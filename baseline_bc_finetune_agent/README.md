# Baseline BC Finetune Agent

**Agent name:** BaselineBCFinetuneAgent

**Author(s):** Shaoyu

## Description

This agent loads a PPO checkpoint produced by
`train_ray_bc_finetune_vs_baseline.py`. It is intended to be initialized from
the behavior-cloned baseline student and then finetuned against the baseline
teacher policy.

After training, copy the selected RLlib checkpoint file into:

```text
baseline_bc_finetune_agent/checkpoint/
```

The expected checkpoint filename looks like:

```text
checkpoint-300
```
