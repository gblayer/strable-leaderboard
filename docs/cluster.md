# Inria's Margaret cluster

Facts collected on 2026-09-30 from `sinfo`, `scontrol` and test jobs.

## Partitions and priority

- Gioia has priority on **parietal**: marg033 to marg035 for CPU and margpu009, margpu014 to margpu016 for GPU.
- On the other partitions, jobs are requeued when a team with priority needs the nodes. The settings are `PreemptMode = REQUEUE` and `PreemptType = preempt/partition_prio`.
- Timed leaderboard runs therefore go on parietal.

| Partition | Nodes | Time limit |
|---|---|---|
| parietal | marg[033-035], margpu[009,014-016] | 7 days |
| gpu-best | margpu[002-009,012-032] | 7 days |
| gpu | margpu[001,010-011] | 7 days |
| normal | marg[001-032] | 7 days |
| normal-best | marg[033-046] | 7 days |
| ci | marg[031-032] | 2 hours |

## GPU nodes on parietal

| Node | GPUs | CPUs | RAM |
|---|---|---|---|
| margpu009 | 2 Ampere | 96 | 1 TB |
| margpu014 | 2 Ampere | 96 | 512 GB |
| margpu015 | 2 Ampere | 96 | 512 GB |
| margpu016 | 2 Ampere | 96 | 512 GB |

margpu014 to margpu016 are identical, so they are the reference nodes. "Ampere" can mean an A100 or an A40, and this shows which:

```
srun -p parietal -w margpu014 --gres=gpu:1 -t 2 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

The Hopper GPUs, margpu012 and margpu013, are faster but outside parietal, so jobs there can be preempted.

## Constraints for an automated runner

- **scrontab is disabled** (`ScronParameters = (null)`). Recurring runs need a login-node cron job, if the admins allow it, or a manual command.
- **Compute nodes appear to have no internet:** `srun -p ci curl -sI https://huggingface.co` printed nothing. Download packages and checkpoints on the login node first, and run jobs with `HF_HUB_OFFLINE=1`.
- Existing jobs use `--partition=gpu-best --constraint=ampere --exclude=margpu005,margpu006,margpu007` and data under `/data/parietal/store4/soda/gblayer`.

## Still to check

- Is Apptainer or Singularity available?
- Can a service account run the leaderboard instead of a personal account?
- Is a cron job on the login node allowed?
