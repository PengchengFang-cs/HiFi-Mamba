import time
import math
import copy
from functools import partial
from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from pytorch_wavelets import DWTForward
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def normalize_feat(x, eps=1e-8):
    # x: [B,C,H,W] (torch)
    # 先按通道去空间均值，削弱 DC
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    # 再按每张特征图做 L2 归一，比较“形状”而非能量
    l2 = torch.sqrt((x**2).sum(dim=(-2, -1), keepdim=True) + eps)
    x = x / l2
    return x

# ---------- 2D FFT -> 幅度频谱（已中心化） ----------
def fft2_magnitude_normed(x, eps=1e-12):
    # x: [B,C,H,W] torch
    B, C, H, W = x.shape
    X = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2,-1)), dim=(-2,-1))
    mag = torch.abs(X) / (H * W)   # 关键：按尺寸归一
    return mag.clamp_min(eps)      # 防 log(0)


# ---------- 将频谱下采样到14x14，以做缩略图 ----------
def shrink_to_14x14(spec):
    # spec: [B,C,H,W] torch
    t = spec.float()
    H, W = t.shape[-2:]
    ph, pw = max(H // 14, 1), max(W // 14, 1)
    if ph > 1 or pw > 1:
        t = F.avg_pool2d(t, kernel_size=(ph, pw), stride=(ph, pw), ceil_mode=True)
    else:
        t = F.interpolate(t, size=(14,14), mode='bilinear', align_corners=False)
    return t


# ---------- 径向平均：把2D频谱按半径bin成1D曲线 ----------
def radial_profile_1d(avg_mag_2d):
    H, W = avg_mag_2d.shape
    cy, cx = (H-1)/2.0, (W-1)/2.0
    yy, xx = np.indices((H, W))
    rr = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    nbins = int(np.ceil(rr.max())) + 1
    prof = np.zeros(nbins, dtype=np.float64)
    cnt  = np.zeros(nbins, dtype=np.int64)
    ridx = np.floor(rr).astype(int)
    np.add.at(prof, ridx, avg_mag_2d)
    np.add.at(cnt,  ridx, 1)
    prof /= np.maximum(cnt, 1)
    r_norm = np.arange(nbins) / (nbins - 1 + 1e-8)
    return r_norm, prof

def plot_radial_curve(spec_b, spec_a, save_path="figB.png", show_delta=False):
    # 输入 torch [B,C,H,W]（未经缩略）
    sb = spec_b.mean(dim=(0,1)).cpu().numpy()  # [H,W]
    sa = spec_a.mean(dim=(0,1)).cpu().numpy()
    r_b, pb = radial_profile_1d(sb)
    r_a, pa = radial_profile_1d(sa)
    yb, ya = np.log(pb + 1e-12), np.log(pa + 1e-12)

    plt.figure(figsize=(5,4))
    plt.plot(r_b, yb, label='before')
    plt.plot(r_a, ya, label='after')
    if show_delta:
        plt.plot(r_b, (ya - yb), label='Δlog(after-before)')
    plt.xlabel('Frequency'); plt.ylabel('log amplitude'); plt.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()


# ---------- 画图A：8个通道的14x14频谱缩略图 ----------
def plot_grid_8(spec14_before, spec14_after, save_path="figA.png"):
    # 输入 torch [B,C,14,14]
    fb = spec14_before.mean(dim=0)  # [C,14,14]
    fa = spec14_after.mean(dim=0)
    C = fb.shape[0]
    idxs = torch.linspace(0, C-1, 8).long()
    tiles_b, tiles_a = fb[idxs], fa[idxs]   # [8,14,14]

    both = torch.cat([tiles_b.flatten(), tiles_a.flatten()])
    vmax = torch.quantile(both, 0.99).item()   # 99% 分位，抗 outlier
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(tiles_b[i].cpu(), cmap='viridis', vmin=0, vmax=vmax); axes[0, i].axis('off')
        axes[1, i].imshow(tiles_a[i].cpu(), cmap='viridis', vmin=0, vmax=vmax); axes[1, i].axis('off')
    axes[0,0].set_ylabel("before"); axes[1,0].set_ylabel("after")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close(fig)

from .mamba_sys_og import SS2D as SS2DSimple

class ChannelAttentionBlock(nn.Module):
    """
    通道注意力模块（Channel Attention Block）
    采用 1x1 卷积 和 3x3 卷积，然后加权输入特征
    """
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1x1(out)
        out = self.relu(out)
        out = self.conv3x3(out)
        
        return self.sigmoid(out) * x  # Hadamard product（逐元素相乘）
                         # [B,H,W,C]

class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(
            dim, 2*dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x= self.norm(x)

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, (dim_scale**2)*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):

        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x= self.norm(x)

        return x
    

class Unpatchify(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.layer = nn.Linear(dim, 2 * dim_scale**2, bias=False)

    def forward(self, x):

        x = self.layer(x)
        _, _, _, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        return x.permute(0, 3, 1, 2)

class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_low = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),

        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        self.x_proj_weight_low = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        kernel_size = 15
        dilation = 2
        padding = dilation * (kernel_size - 1) // 2  # 计算为 4

        self.conv_dt = nn.Conv1d(
            in_channels=self.dt_rank,
            out_channels=self.dt_rank,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=self.dt_rank,
            bias=False
        )

        self.conv_B = nn.Conv1d(
            in_channels=self.d_state,
            out_channels=self.d_state,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=self.d_state,
            bias=False
        )

        self.conv_C = nn.Conv1d(
            in_channels=self.d_state,
            out_channels=self.d_state,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=self.d_state,
            bias=False
        )

        self.simple_gate_b = SimpleGate1D(self.d_state)
        self.simple_gate_c = SimpleGate1D(self.d_state)
        # self.simple_gate_dts = SimpleGate(self.dt_rank)

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )

        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True) # (K=4, D, N)

        self.selective_scan = selective_scan_fn


        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, high_branch_low: torch.Tensor):
        # self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        # print("1. x shape:", x.shape, "high_branch_low shape:", high_branch_low.shape)
        L = H * W
        K = 1
        x_hwwh = x.view(B, 1, -1, L)
        xs = x_hwwh
        #####################################
        high_branch_low_seq = high_branch_low.view(B, 1, -1, L)
        low_dbl = torch.einsum("b k d l, k c d -> b k c l", high_branch_low_seq.view(B, K, -1, L), self.x_proj_weight_low)
        dts_low, Bs_low, Cs_low = torch.split(low_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
    
        Bs_low = self.simple_gate_b(Bs_low)
        Cs_low = self.simple_gate_c(Cs_low)
        # dts_low = self.simple_gate_c(Cs_low)
      
        #####################################
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = self.x_conv(x_dbl.squeeze(1)).unsqueeze(1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
    
        #######################################
        Bs = Bs + Bs_low
        Cs = Cs + Cs_low

        dts = self.conv_dt(dts.squeeze(1)).unsqueeze(1)
        Bs = self.conv_B(Bs.squeeze(1)).unsqueeze(1)
        Cs = self.conv_C(Cs.squeeze(1)).unsqueeze(1)

        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape, Bs.shape, Cs.shape, Ds.shape, dts.shape)

        #print("As", As.shape, "Bs", Bs.shape, "Cs", Cs.shape, "Ds", Ds.shape, "dts", dts.shape, "dt_projs_bias", dt_projs_bias.shape)
        # As torch.Size([64, 8]) Bs torch.Size([2, 1, 8, 65536]) Cs torch.Size([2, 1, 8, 65536]) Ds torch.Size([64]) dts torch.Size([2, 64, 65536]) dt_projs_bias torch.Size([64])
        # As torch.Size([512, 16]) Bs torch.Size([2, 1, 16, 4096]) Cs torch.Size([2, 1, 16, 4096]) Ds torch.Size([512]) dts torch.Size([2, 128, 4096]) dt_projs_bias torch.Size([128])
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]
    

    def forward(self, x: torch.Tensor, high_branch_low: torch.Tensor,**kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)).contiguous()# (b, d, h, w)

        high_branch_low = self.in_proj_low(high_branch_low).permute(0, 3, 1, 2).contiguous()
 
        y = self.forward_core(x, high_branch_low)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class DWInvertedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio=5, stride=1):
        super(DWInvertedBottleneck, self).__init__()
        hidden_dim = in_channels * expansion_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
    
        self.block = nn.Sequential(
            # 1. Pointwise conv (expand)
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),

            # 2. Depthwise conv
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),

            # 3. Pointwise conv (project)
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.use_residual:
            h = self.block(x)
            return x + h
        else:
            return self.block(x)



# ===== 核生成 =====
def binomial_1d_5(dtype=torch.float64, device=None):
    k = torch.tensor([1., 4., 6., 4., 1.], dtype=dtype, device=device)
    k /= k.sum()
    return k

def gaussian_1d(k, sigma, dtype=torch.float64, device=None):
    assert k % 2 == 1
    ax = torch.arange(k, dtype=dtype, device=device) - k // 2
    g = torch.exp(-(ax**2) / (2.0 * sigma**2))
    g /= g.sum()
    return g

# ===== 分离卷积工具 =====
def _sep_conv(x, k1d, horizontal=True, stride_hw=(1,1)):
    C = x.size(1)
    if horizontal:
        w = k1d.view(1, 1, 1, -1)
    else:
        w = k1d.view(1, 1, -1, 1)
    w = w.to(dtype=x.dtype, device=x.device).expand(C, 1, *w.shape[-2:])
    return F.conv2d(x, w, stride=stride_hw, groups=C)

# ===== Reduce（低通+下采样） =====
class SeparableReduce(nn.Module):
    def __init__(self, kernel_size=5, use_binomial=True, sigma=None, padding_mode="reflect"):
        super().__init__()
        self.pad = kernel_size // 2
        self.padding_mode = padding_mode
        if use_binomial:
            assert kernel_size == 5
            k1d = binomial_1d_5()
        else:
            assert sigma is not None
            k1d = gaussian_1d(kernel_size, float(sigma))
        self.register_buffer("k1d", k1d)  # 高精度保存

    def forward(self, x):
        x = F.pad(x, (self.pad,)*4, mode=self.padding_mode)
        x = _sep_conv(x, self.k1d, horizontal=True, stride_hw=(1,1))
        x = _sep_conv(x, self.k1d, horizontal=False, stride_hw=(2,2))
        return x

# ===== Expand（上采样+低通+补偿） =====
class SeparableExpand(nn.Module):
    def __init__(self, kernel_size=5, use_binomial=True, sigma=None, padding_mode="reflect"):
        super().__init__()
        self.pad = kernel_size // 2
        self.padding_mode = padding_mode
        if use_binomial:
            assert kernel_size == 5
            k1d = binomial_1d_5()
        else:
            assert sigma is not None
            k1d = gaussian_1d(kernel_size, float(sigma))
        self.register_buffer("k1d", k1d)

    @staticmethod
    def _zero_insert_2x(x):
        N, C, H, W = x.shape
        y = x.new_zeros(N, C, H*2, W*2)
        y[:, :, ::2, ::2] = x
        return y

    def forward(self, x):
        x = self._zero_insert_2x(x)
        x = F.pad(x, (self.pad,)*4, mode=self.padding_mode)
        x = _sep_conv(x, self.k1d, horizontal=True, stride_hw=(1,1))
        x = _sep_conv(x, self.k1d, horizontal=False, stride_hw=(1,1))
        return x * 4  # up=2 => 乘以4补偿能量

# ===== 拉普拉斯金字塔（单层） =====
# def laplacian_pyramid(x, reducer, expander):
#     lo = reducer(x)             # 低频
#     lo_up = expander(lo)         # 上采样回原分辨率
#     hi = x - lo_up               # 高频
#     return lo, hi

class WLBlock(nn.Module):
    def __init__(self, dim):
        super(WLBlock, self).__init__()

        self.conv = DWInvertedBottleneck(in_channels=dim, out_channels=dim, expansion_ratio=5)
        # self.dwt = DWTForward(J=1, wave='haar', mode='zero')  
        self.reduce = SeparableReduce(kernel_size=5, use_binomial=True)
        # self.expand = SeparableExpand(kernel_size=5, use_binomial=True)

    def laplacian_pyramid(self, x, reducer, expander=None):
        lo = reducer(x)             # 低频
        # lo_up = expander(lo)         # 上采样回原分辨率
        # hi = x - lo_up               # 高频
        return lo

    def forward(self, x):
        x = self.conv(x)
        low = self.laplacian_pyramid(x, self.reduce)
        low = F.interpolate(low, scale_factor=2, mode='bilinear')  # 恢复到原始大小
        high = x - low  # 计算高频部分
        # low, _ = self.dwt(x) 
        # low = F.interpolate(low, scale_factor=2, mode='bilinear')
        # high = x - low

        return low, high

class SimpleGate1D(nn.Module):
    def __init__(self, dim, expansion_factor=6, bias=False):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.fc1 = nn.Linear(dim, hidden_dim * 2, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):  # shape: [B, *, dim]
        orig_shape = x.shape
        if x.dim() == 4:
            # x: [B, K, C, L] → [B*K*L, C]
            B, K, C, L = x.shape
            x = x.permute(0, 1, 3, 2).contiguous().view(-1, C)
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = F.gelu(x1) * x2
        x = self.fc2(x)

        if len(orig_shape) == 4:
            # 恢复成 [B, K, C, L]
            x = x.view(B, K, L, -1).permute(0, 1, 3, 2).contiguous()

        return x

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)
    
def GLUFeedForward(dim, mult=2, dropout=0.):
    dim_hidden = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_hidden * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim)
    )

# ---------------- 按通道的 MLP (最后一维) ----------------
class ChannelMLP(nn.Module):
    def __init__(self, dim, multi=2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim*multi)
        # self.act = F.gelu()
        self.fc2 = nn.Linear(dim*multi, dim)

    def forward(self, x):
        # x: [B,H,W,C]
        return self.fc2(F.gelu(self.fc1(x)))

# ---------------- MoE (BHWC, shared + routed, per-pixel routing) ----------------
class SharedRoutedMoE_BHWC(nn.Module):
    def __init__(self, dim, shared=1, routed=2, top_k=1, hidden_mult=2):
        """
        dim: C
        shared: 共享专家个数（总是参与）
        routed: 路由专家个数（由 gate 选择）
        top_k: 选择的专家数（轻量建议 1）
        hidden_mult: 隐藏维 = hidden_mult * C（轻量建议 2）
        """
        super().__init__()
        hidden_dim = max(dim * hidden_mult, dim)

        # 共享专家（始终激活）
        self.shared = nn.ModuleList([ChannelMLP(dim) for _ in range(shared)])
        # 路由专家（按 gate 选择）
        self.experts = nn.ModuleList([ChannelMLP(dim) for _ in range(routed)])

        # 路由器（按通道特征在每个像素点生成 logits）
        self.router = nn.Linear(dim, routed)
        self.top_k = top_k

        # 记录负载均衡项
        self.balance_loss = torch.tensor(0.0)

        # ==== 新增：统计用 ====
        self.num_experts = routed
        self.register_buffer("usage_counts", torch.zeros(self.num_experts))  # 累加每个 expert 使用次数
        self.register_buffer("total_tokens", torch.zeros(1))                 # 累加总 token 数
        self.is_moe = True 

    def _balance(self, gates: torch.Tensor):
        """
        gates: [B, H, W, E]  —— softmax 之后的 gate 概率
        返回：标量 balance loss，值越小表示越均匀
        """
        # 维度
        B, H, W, E = gates.shape

        # 1. importance：每个 expert 的平均 gate 概率（soft）
        #    越大说明 router 越“偏爱”这个 expert
        importance = gates.mean(dim=(0, 1, 2))          # [E]

        # 2. load：每个 expert 实际被分配到的 token 比例（hard）
        #    使用 argmax 得到 top-1 expert，再 one-hot 统计
        top_idx = gates.argmax(dim=-1)                  # [B, H, W]
        one_hot = F.one_hot(top_idx, num_classes=E)     # [B, H, W, E]
        load = one_hot.float().mean(dim=(0, 1, 2))      # [E]

        # 3. 计算 balance loss
        #    按 Switch Transformer 的风格：E * sum(importance * load)
        #    两个向量都越接近均匀分布，这个值就越小
        loss = (importance * load).sum() * E            # 标量

        return loss

    @torch.no_grad()
    def _update_usage(self, top_idx: torch.Tensor):
        """
        top_idx: [B,H,W]，每个 token 选中的 expert id（0~E-1）
        """
        # 展平
        idx_flat = top_idx.reshape(-1)  # [N]
        # 统计每个 expert 被选中的次数
        counts = torch.bincount(idx_flat,
                                minlength=self.num_experts).to(self.usage_counts.dtype)  # [E]

        # 累加到 buffer
        self.usage_counts += counts
        self.total_tokens += idx_flat.numel()

    def get_usage(self):
        """
        返回每个 expert 的使用比例（0~1），总和为 1。
        如果还没跑过 forward，返回 None。
        """
        if self.total_tokens.item() == 0:
            return None
        usage = self.usage_counts / self.total_tokens
        return usage

    def forward(self, x):
        # x: [B,H,W,C]
        B,H,W,C = x.shape

        # 共享专家输出：逐像素同一套 MLP
        shared_out = 0.0
        for exp in self.shared:
            shared_out = shared_out + exp(x)                          # [B,H,W,C]

        # 路由 logits & softmax（逐像素）
        logits = self.router(x)                                       # [B,H,W,E]
        gates = F.softmax(logits, dim=-1)                             # [B,H,W,E]

        # 记录 balance loss（给训练 loop 使用）
        self.balance_loss = self._balance(gates)

        if self.top_k == 1:
            B, H, W, C = x.shape
            num_experts = len(self.experts)

            # 1. 计算每个 token 选中的 expert id
            # gates: [B,H,W,E]
            top_idx = gates.argmax(dim=-1)              # [B,H,W]

            # ==== 新增：更新 usage 统计 ====
            self._update_usage(top_idx)

            # 2. 展平 token，方便按 expert 路由
            x_flat        = x.reshape(B * H * W, C)           # [N, C]
            top_idx_flat  = top_idx.reshape(-1)               # [N]
            gates_flat    = gates.reshape(-1, num_experts)    # [N, E]  ← 这里就是 gates_flat

            # 3. 预先分配输出缓冲区
            routed_flat = torch.zeros_like(x_flat)      # [N, C]

            # 4. 按 expert 分组运行，只算被分配到的 token
            for eid, expert in enumerate(self.experts):
                mask = (top_idx_flat == eid)            # [N] bool
                if not mask.any():
                    continue

                # 取出属于该专家的 tokens
                x_e = x_flat[mask]                      # [N_e, C]

                # 对应的 gate 权重（top-1，只取该 expert 的一列）
                gate_e = gates_flat[mask, eid:eid+1]    # [N_e, 1]

                # 只在这 N_e 个 token 上跑一次该 expert
                y_e = expert(x_e)                       # [N_e, C]

                # 乘上 gate 权重（更接近 Switch 的做法）
                y_e = gate_e * y_e                      # [N_e, C]

                # 将结果 scatter 回全局位置
                routed_flat[mask] = y_e

            # 5. reshape 回去
            routed_out = routed_flat.reshape(B, H, W, C)


        return shared_out + routed_out, self.balance_loss             # [B,H,W,C], scalar

class SEBlockBHWC(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B,H,W,C]
        B,H,W,C = x.shape
        # 全局平均池化 (沿 H,W) -> [B,C]
        y = x.mean(dim=(1,2))                      # [B,C]
        gate = self.fc(y).view(B,1,1,C)            # [B,1,1,C]
        return x * gate   

class CrossFreqConsensusRouting(nn.Module):
    """
    Cross-Frequency Consensus Routing (CFCR)
    - 输入: high, high_hidden [B, C, H, W]
    - 输出: fused_high [B, C, H, W]
    - 额外参数: 只引入 1 个可学习标量 gamma
    """
    def __init__(self, init_gamma: float = 2.0, eps: float = 1e-6):
        super().__init__()
        # 用 log_gamma 保证 gamma > 0，方便训练
        self.log_gamma = nn.Parameter(torch.log(torch.tensor(init_gamma)))
        self.eps = eps

    def forward(self, high, high_hidden):
        # high, high_hidden: [B, C, H, W]
        # 1. 计算 channel-wise cosine similarity
        #    cos \in [-1,1], 尺度为 [B,1,H,W]
        num = (high * high_hidden).sum(dim=1, keepdim=True)  # 点乘
        den = (high.norm(dim=1, keepdim=True) * 
               high_hidden.norm(dim=1, keepdim=True) + self.eps)
        cos = num / den

        # 2. 映射到 gate: w = sigmoid(gamma * cos)
        gamma = self.log_gamma.exp()
        w = torch.sigmoid(gamma * cos)  # [B,1,H,W]

        # 3. 融合: 高频共识路由
        fused = w * high_hidden + (1.0 - w) * high
        return fused
    
class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        id = 0,
        expert_mult = 4,
        **kwargs,
    ):
        super().__init__()
        # self.local = LocalBlock(dim=hidden_dim, hidden_dim=hidden_dim *4)

        hidden_dim = hidden_dim // 2
        self.ln_global = norm_layer(hidden_dim*2)
        self.ln_1 = norm_layer(hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        
        self.cfcr = CrossFreqConsensusRouting()

        # MOE Part
        # self.self_attention_global = SS2DSimple(d_model=hidden_dim*2, dropout=attn_drop_rate, d_state=d_state, **kwargs) # 初始化我们的global分支的mamba
        self.self_attention_global = SEBlockBHWC(hidden_dim*2, reduction=8) # x是bchw
        self.route_score = nn.Conv2d(2 * hidden_dim, 2, kernel_size=1)
        # Experts for the first level
        self.moe = SharedRoutedMoE_BHWC(
            dim=hidden_dim*2,
            shared=1,
            routed=4,
            top_k=1,
            hidden_mult=2
        )

        # Mamba Part
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.wl_block = WLBlock(hidden_dim)
        self.conv0 = DWInvertedBottleneck(in_channels=hidden_dim, out_channels=hidden_dim, expansion_ratio=5)
        self.conv1 = DWInvertedBottleneck(in_channels=hidden_dim, out_channels=hidden_dim, expansion_ratio=5)
        # print('hidden_dimhidden_dim', hidden_dim*2)
        self.moe_pre = nn.Linear(hidden_dim*4, hidden_dim*2) 
        self.moe_post = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.channel_attention = ChannelAttentionBlock(hidden_dim*2)
        self.idx = id
        
    def _conv_bhwc(self, x, conv: nn.Conv2d):
        # 辅助：BHWC <-> NCHW
        B,H,W,C = x.shape
        x_nchw = x.permute(0,3,1,2).contiguous()
        y = conv(x_nchw)
        return y.permute(0,2,3,1).contiguous()
    
    def forward(self, input: torch.Tensor):
        B, H, W, C = input.shape
   
        # 分支create
        global_features = input
        global_features = self.self_attention_global(self.ln_global(global_features))
        high, low = torch.split(input, C // 2, dim=-1)


        # 下面的部分作为moe的其中两个分支，做一个route的灵活权重配置
        high = high.permute(0, 3, 1, 2)
        low = low.permute(0, 3, 1, 2)

        low, high_hidden_feautres = self.wl_block(low)

        # high分支融合
        high = high + high_hidden_feautres
        high = self.cfcr(high, high_hidden_feautres)

        # 后续分支
        high = self.conv0(high)
        output_high = self.conv1(high).permute(0,2,3,1)
        
        low = low.permute(0, 2, 3, 1)
        high = high.permute(0, 2, 3, 1)

        low = self.self_attention(self.ln_1(low), self.ln_2(high))

        # 同路
        x = torch.cat([global_features, output_high, low], dim=-1)
        x = self.moe_pre(x)
        x, balance_loss = self.moe(x)
        x = self.moe_post(x)

        x = x.permute(0, 3, 1, 2)
        x = self.channel_attention(x)
        x = self.drop_path(x)

        x = x.permute(0, 2, 3, 1) + input
        
        return x, balance_loss


def ifft2c(x, dim=((-2,-1)), img_shape=None):
    x = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=dim), s=img_shape, dim=dim), dim = dim)
    return x

def fft2c(x, dim=((-2,-1)), img_shape=None):
    x = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=dim), s=img_shape, dim=dim), dim = dim)
    return x

class DataConsistency(nn.Module):
    def __init__(self, num_of_feat_maps, patchify, patch_size=2):
        super(DataConsistency, self).__init__()
        self.unpatchify = Unpatchify(num_of_feat_maps, dim_scale=patch_size)
        self.activation = nn.SiLU()
        self.patchify = patchify
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=2, embed_dim=num_of_feat_maps, norm_layer=nn.LayerNorm)

    def data_cons_layer(self, im, mask, zero_fill, coil_map):
        im_complex = im[:,0,:,:] + 1j * im[:,1,:,:]
        zero_fill_complex = zero_fill[:,0,:,:] + 1j * zero_fill[:,1,:,:]
        zero_fill_complex_coil_sep = torch.tile(zero_fill_complex.unsqueeze(1), dims=[1,coil_map.shape[1],1,1]) * coil_map
        im_complex_coil_sep = torch.tile(im_complex.unsqueeze(1), dims=[1,coil_map.shape[1],1,1]) * coil_map
        actual_kspace = fft2c(zero_fill_complex_coil_sep)
        gen_kspace = fft2c(im_complex_coil_sep)
        mask_bool = mask>0
        mask_coil_sep = torch.tile(mask_bool, dims=[1,coil_map.shape[1],1,1])
        gen_kspace_dc = torch.where(mask_coil_sep, actual_kspace, gen_kspace)
        gen_im = torch.sum(ifft2c(gen_kspace_dc) * torch.conj(coil_map), dim=1)
        gen_im_return = torch.stack([torch.real(gen_im), torch.imag(gen_im)], dim=1)
        return gen_im_return.type(im.dtype)
    
    def forward(self, x, zero_fill, mask, coil_map):
        h = self.unpatchify(x)  

        h = self.data_cons_layer(h, mask, zero_fill, coil_map)
        if self.patchify:
            h = self.activation(h)
            h = self.patch_embed(h)
            return x + h, 0
        else:
            return h, 0


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                id=i,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x, us_im=None, us_mask=None, coil_maps=None):
        balance_loss_total = 0
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x, balance_loss = blk(x)
                balance_loss_total += balance_loss
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x, balance_loss_total / len(self.blocks)

class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        Upsample (nn.Module | None, optional): Upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        upsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = PatchExpand(dim, dim_scale=2, norm_layer=nn.LayerNorm)
        else:
            self.upsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.upsample is not None:
            x = self.upsample(x)

        return x

class VSSM_unrolled(nn.Module): 
    def __init__(self, patch_size=4, in_chans=2, num_classes=2, depths=[2, 2, 2, 2, 2, 2], 
                 hidden_dim=128, d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()
        dims=[hidden_dim, hidden_dim, hidden_dim, hidden_dim]
        print("able", depths)
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.num_features_up = int(dims[0] * 2)
        self.dims = dims
        self.final_upsample = final_upsample

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim = hidden_dim,
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample= None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
            self.layers.append(DataConsistency(hidden_dim, patchify=True, patch_size=patch_size))

        self.last_dc = DataConsistency(hidden_dim, patchify=False, patch_size=patch_size)
        self.norm = norm_layer(self.num_features)


        self.apply(self._init_weights)



    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        # print(m, getattr(getattr(m, "weight", nn.Identity()), "INIT", None), isinstance(m, nn.Linear), "======================")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    #Encoder and Bottleneck
    def forward_features(self, x, us_im, us_mask, coil_map):
        x = self.patch_embed(x)
        balance_loss_total = 0

        for layer in self.layers:

            x, balance_loss = layer(x, us_im, us_mask, coil_map)
            balance_loss_total += balance_loss

        x = self.norm(x)  # B H W C

        return x, balance_loss_total / len(self.layers)


    def forward(self, x, us_mask, coil_map):
        """
        x shape b2hw
        b1hw
        """
        us_im = x.clone()
        x, balance_loss = self.forward_features(x, us_im, us_mask, coil_map)
        x, _ = self.last_dc(x, us_im, us_mask, coil_map)
        return x, balance_loss


    def flops(self, shape=(2, 256, 256)):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit, # latter
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input1 = torch.randn((1, 2, 256, 256), device=next(model.parameters()).device)
        input2 = torch.randn((1, 5, 256, 256), device=next(model.parameters()).device)
        input3 = torch.randn((1, 1, 256, 256), device=next(model.parameters()).device)

        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input1, input2, input3), supported_ops=supported_ops)

        del model, input1, input2, input3
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"



class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=2, num_classes=2, depths=[2, 2, 2, 2], 
                 dims=[96, 192, 384, 768], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.num_features_up = int(dims[0] * 2)
        self.dims = dims
        self.final_upsample = final_upsample
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                # dim=dims[i_layer], #int(embed_dim * 2 ** i_layer)
                dim = int(dims[0] * 2 ** i_layer),
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(dims[0]*2**(self.num_layers-1-i_layer)),
            int(dims[0]*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer ==0 :
                layer_up = PatchExpand(dim=int(self.embed_dim * 2 ** (self.num_layers-1-i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = VSSLayer_up(
                    dim= int(dims[0] * 2 ** (self.num_layers-1-i_layer)),
                    depth=depths[(self.num_layers-1-i_layer)],
                    d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                    drop=drop_rate, 
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(dim_scale=patch_size,dim=self.embed_dim)
            self.output = nn.Conv2d(in_channels=self.embed_dim,out_channels=self.num_classes,kernel_size=1,bias=False)

        self.apply(self._init_weights)



    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        # print(m, getattr(getattr(m, "weight", nn.Identity()), "INIT", None), isinstance(m, nn.Linear), "======================")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    #Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)

        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)
        x = self.norm(x)  # B H W C
        return x, x_downsample

    #Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x,x_downsample[3-inx]],-1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B H W C
  
        return x
    def up_x4(self, x, patch_size):
        if self.final_upsample=="expand_first":
            B,H,W,C = x.shape
            x = self.up(x)
            x = x.view(B, patch_size*H, patch_size*W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)
            
        return x
    def forward(self, x):
        x,x_downsample = self.forward_features(x)
        x = self.forward_up_features(x,x_downsample)
        x = self.up_x4(x, self.patch_size)
        return x


    def flops(self, shape=(3, 224, 224)):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit, # latter
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"


# APIs with VMamba2Dp =================
def check_vssm_equals_vmambadp():
    from bak.vmamba_bak1 import VMamba2Dp

    # test 1 True =================================
    torch.manual_seed(time.time()); torch.cuda.manual_seed(time.time())
    oldvss = VMamba2Dp(depths=[2,2,6,2]).half().cuda()
    newvss = VSSM(depths=[2,2,6,2]).half().cuda()
    newvss.load_state_dict(oldvss.state_dict())
    input = torch.randn((12, 3, 224, 224)).half().cuda()
    torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y1 = oldvss.forward_backbone(input)
    torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y2 = newvss.forward_backbone(input)
    print((y1 -y2).abs().sum()) # tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
    
    # test 2 True ==========================================
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    oldvss = VMamba2Dp(depths=[2,2,6,2]).cuda()
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    newvss = VSSM(depths=[2,2,6,2]).cuda()

    miss_align = 0
    for k, v in oldvss.state_dict().items(): 
        same = (oldvss.state_dict()[k] == newvss.state_dict()[k]).all()
        if not same:
            print(k, same)
            miss_align += 1
    print("init miss align", miss_align) # init miss align 0

if __name__ == "__main__":

    layers = VSSM_unrolled().cuda()
    x = torch.randn((2, 2, 320, 320)).cuda()
    us_im = torch.randn((2, 1, 320, 320)).to(torch.complex64).cuda()
    us_mask = torch.randn((2, 1, 320, 320)).cuda()
    coil_map = torch.ones((2, 1, 320, 320)).cuda()
    y = layers(x, us_mask, coil_map)
    print(y.shape) # torch.Size([2, 2, 256, 256])



