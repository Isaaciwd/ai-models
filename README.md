# ai-models

`ai-models` is a command-line runner for AI weather models.

It provides one consistent CLI across models, plus a YAML-first workflow so you can run with a single command:

```bash
ai-models --yaml /path/to/run.yaml
```

Model implementations are installed as plugins (for example `ai-models-fourcastnetv2`).

## Install

Install the core package:

```bash
pip install ai-models
```

Install at least one model plugin (example):

```bash
pip install ai-models-fourcastnetv2
```

List installed models:

```bash
ai-models --models
```

## Quick Start (YAML)

Create `run.yaml`:

```yaml
model: fourcastnetv2-small

run:
  input: cds
  output: none
  date: 20230110
  time: 0000
  lead_time: 24
  only_gpu: true

runtime:
  assets_dir: ./assets/fourcastnetv2-small
```

Run:

```bash
ai-models --yaml ./run.yaml
```

## Quick Start (CLI)

You can still run directly from flags:

```bash
ai-models --input cds --date 20230110 --time 0000 --lead-time 24 \
  --output none --assets ./assets/fourcastnetv2-small fourcastnetv2-small
```

## YAML Mapping

`--yaml` maps YAML keys to normal CLI options.

Common keys:

- `model` -> positional `MODEL`
- `run.input` -> `--input`
- `run.output` -> `--output`
- `run.date` -> `--date`
- `run.time` -> `--time`
- `run.lead_time` -> `--lead-time`
- `run.only_gpu` -> `--only-gpu`
- `runtime.assets_dir` -> `--assets`

Advanced mapping is available through optional `cli` keys.

Examples:

- `cli.debug` -> `--debug`
- `cli.verbose` -> `--verbose`
- `cli.num_threads` -> `--num-threads`
- `cli.model_args` -> extra model-plugin arguments

## Input Date and Time Formats

Accepted `run.date` formats:

- `YYYYMMDD` (example: `20230110`)
- `YYYY-MM-DD` (example: `2023-01-10`)
- relative integer day offsets (example: `-1` for yesterday, `0` for today)

Accepted `run.time` formats:

- `HHMM` (example: `0000`, `1200`)
- `HH:MM` (example: `00:00`, `12:00`)

## Path Behavior in YAML

For path-like YAML values (such as assets and file paths):

- relative paths are resolved relative to the YAML file directory
- placeholders are supported:
  - `{repo_root}`
  - `{yaml_dir}`
  - `{cwd}`

Example:

```yaml
runtime:
  assets_dir: "{yaml_dir}/assets/fourcastnetv2-small"
```

## Sensitivity in the Same YAML

For models that support sensitivity (like `fourcastnetv2-small`), keep everything in one YAML file.

Example:

```yaml
model: fourcastnetv2-small

run:
  input: cds
  output: none
  date: 20230110
  time: 0000
  lead_time: 24

runtime:
  assets_dir: ./assets/fourcastnetv2-small

output:
  path: ./sensitivity-results.nc
  summary_path: ./sensitivity-results.json

plotting:
  enabled: true
  prefix: ./sensitivity-results
  top_k: 6

targets:
  - name: west-coast-r850
    param: r
    level: 850
    area: [50, 230, 30, 245]
    metric: mean-square
```

If `targets`, `plotting`, or sensitivity `output` keys are present, `ai-models --yaml ...` automatically enables `--sensitivity-config` using that same YAML.

## Precedence Rules

- explicit CLI flags win over YAML values
- YAML values win over built-in defaults

This lets you keep reproducible configs while still doing quick one-off overrides.

## Common Commands

List models:

```bash
ai-models --models
```

Show required input fields for a model:

```bash
ai-models --fields fourcastnetv2-small
```

Print retrieve requests only:

```bash
ai-models --yaml ./run.yaml --retrieve-requests
```

Download model assets:

```bash
ai-models --download-assets --assets ./assets/fourcastnetv2-small fourcastnetv2-small
```

## Troubleshooting

- `Unknown model ...`: install the model plugin and run `ai-models --models`
- `ai-models: command not found`: verify your environment activation and install
- missing asset files: run `--download-assets` or point `runtime.assets_dir` to the correct path

## License

Apache License 2.0. See `LICENSE`.
