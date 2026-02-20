# Agent Guidelines

## Environment
- Use `uv`
- Compile papers using `cd dir_name & ./compile.sh filename`. Use subagents.

## Cluster (AMD HPC Fund)
- SSH alias: `amd`
- Submit jobs: `bash cluster/submit.sh cluster/<script>.slurm`
- SLURM fails silently (signal 53) if `logs/` directory doesn't exist
- PyTorch ROCm: `uv sync` then `uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/rocm7.1`
- Can't use `[tool.uv.sources]` for ROCm torch due to `triton-rocm` cross-platform resolution issues

## Style
- Don't put + in front of coefficients

## Writing Rules
- No em dashes
- Use colons sparingly
- Avoid "framework" - presumptuous
- Avoid listy parentheticals; use prose instead
- Don't start sentences with gerunds
- Don't start sentences with noun phrases
- Use active phrasing: subject + verb
- Connect sentences logically instead of just listing
- Vary sentence rhythm - avoid choppy short sentences
- Avoid repetitive sentence starters
- Avoid using the same connector repeatedly (e.g., "though" three times)
