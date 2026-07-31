# V2 Legacy Archive (2026-04-19)

## Scope

This archive stores V2-only artifacts removed from the active code path:

- V2/V2.1/V2.2/V2.3 denoiser implementations
- V2-related config files
- V2-only inference/debug scripts
- V2 architecture images and docs under diffusion

All files are preserved under:

`archive/v2_legacy_2026-04-19/snapshot/`

The folder keeps the original relative paths to support one-command restore.

## Restore

Run from repository root:

```bash
rsync -av archive/v2_legacy_2026-04-19/snapshot/ ./
```

## Notes

- Current mainline no longer supports V2 runtime switches.
- Restoring these files is intended for historical comparison or rollback experiments.
