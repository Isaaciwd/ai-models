# ai-models sensitivity module

This fork adds a shared sensitivity utility module at:

- `src/ai_models/sensitivity.py`

The goal is to keep user-facing sensitivity configuration/output/plotting logic in `ai-models` while model plugins keep only model-specific differentiation code.

## Shared responsibilities

- sensitivity CLI argument definitions
- YAML config loading and validation
- target metadata structures
- NetCDF and JSON output serialization
- sensitivity plotting (including optional coastlines and signed-gradient maps)

## Plugin responsibilities

- autodiff-capable model rollout
- variable/level/channel mapping per model
- model-specific objective tensor construction
- returning input-state gradients

## Rationale

This architecture avoids duplicating terminal UX and plotting in each model plugin and is designed to simplify adding support for additional differentiable models.
