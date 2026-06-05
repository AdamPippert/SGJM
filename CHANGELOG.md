# Changelog

All notable changes to SGJM are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions are date-based (CalVer, `YYYY.M.D`).

## [2026.6.5] - 2026-06-05

Initial public pre-release.

### Added
- Apache License 2.0 (`LICENSE`) and `NOTICE`.
- Hybrid Mamba-2 / attention backbone with post-gate work.
- Speculative decoding pipeline: parallel draft generation, latent-space branch
  scoring (JEPA), and discriminative verification in a single trainable system.
- Training backends for MLX (Apple Silicon) and PyTorch (CUDA / ROCm / CPU).
- Project metadata, classifiers, and URLs in `pyproject.toml`.

### Notes
- Alpha-stage research prototype. Interfaces, checkpoints, and training recipes
  may change without notice.

[2026.6.5]: https://github.com/AdamPippert/SGJM/releases/tag/2026.6.5
