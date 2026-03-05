# Version Mapping

## Source Mapping

- `versions/v1` <- `/iridisfs/scratch/pf2m24/projects/MRIRecon/MambaRecon/code`
- `versions/v2` <- `/iridisfs/scratch/pf2m24/projects/MRIRecon/MambaReconV2/code`
- `versions/v3` <- `/iridisfs/scratch/pf2m24/projects/MRIRecon/MambaReconV3/code`

## Refactor Decisions

- Keep each version self-contained under `versions/vX/code` to preserve original import behavior.
- Share third-party dependency source under `third_party/` instead of duplicating per version.
- Remove non-code artifacts (logs/images/checkpoints/results) from repository scope.
- Keep launcher scripts in `scripts/` to normalize train/infer execution.
