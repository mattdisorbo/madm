# MADM

## Local setup

```bash
uv sync
```

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/neural_activation_auditor.py` | Compares base vs auditor reasoning paths on loan decisions, trains an SAE on activations, and tests activation steering |

## AMD HPC Fund cluster

### Access

Generate an SSH key and send the public key to hpc.fund@amd.com:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_amd -N "" -C "<username>@amd-cluster"
cat ~/.ssh/id_ed25519_amd.pub | pbcopy
```

Add to `~/.ssh/config`:

```
Host amd
    HostName hpcfund.amd.com
    User <username>
    IdentityFile ~/.ssh/id_ed25519_amd
```

Once the cluster team confirms, test with `ssh amd`.

### First-time setup

From your local machine, sync code:

```bash
rsync -av --exclude .venv --exclude outputs --exclude .git . amd:'$WORK/madm/'
```

Then on the cluster:

```bash
ssh amd
cd "$WORK/madm"
bash cluster/setup.sh
```

### Submitting jobs (from local)

```bash
bash cluster/submit.sh cluster/neural_activation_auditor.slurm
```

The submit script syncs your local code to the cluster, then submits the job.

### Monitoring

```bash
ssh amd 'squeue -u $USER'                        # job status
ssh amd 'sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed'  # job result
ssh amd 'cat $WORK/madm/logs/<name>.<jobid>.out'  # stdout
ssh amd 'cat $WORK/madm/logs/<name>.<jobid>.err'  # stderr
ssh amd 'scancel <jobid>'                         # cancel
```

### Interactive testing

```bash
ssh amd
salloc -N 1 -n 1 -p devel -t 00:30:00
source "$WORK/madm/.venv/bin/activate"
python scripts/neural_activation_auditor.py
```

### Available partitions

| Partition | GPUs | VRAM | FP16 TFLOPS | Max time | Good for |
|-----------|------|------|-------------|----------|----------|
| `devel` | 1x MI210 | 64GB | ~181 | 30 min | Small models, debugging |
| `mi2104x` | 4x MI210 | 64GB | ~181 | 24 hr | Long runs, small models |
| `mi2508x` | 8x MI250 | 128GB | ~362 | 12 hr | Mid-size models |
| `mi3001x` | 1x MI300X | 192GB | ~653 | 4 hr | Large models (70B+) |
| `mi3008x` | 8x MI300X | 192GB | ~653 | 12 hr | Multi-GPU large runs |
| `mi3258x` | 8x MI325X | 256GB | ~653 | 12 hr | Largest models |
