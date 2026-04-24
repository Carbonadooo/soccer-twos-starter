# Baseline BC Finetune Shaped Agent

**Agent name:** BaselineBCFinetuneShapedAgent

**Author(s):** Shaoyu, Haoran

## Description

This agent loads a PPO checkpoint produced by
`train_ray_bc_finetune_vs_baseline_shaped.py`. It is initialized from the
behavior-cloned baseline student and then finetuned against the baseline
teacher with additional reward shaping.

After training, copy the selected RLlib checkpoint file into:

```text
baseline_bc_finetune_shaped_agent/checkpoint/
```

The expected checkpoint filename looks like:

```text
checkpoint-300
```
