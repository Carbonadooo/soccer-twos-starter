# Baseline Behavior Cloning Agent

**Agent name:** BaselineBCPlayer

**Author(s):** Shaoyu

## Description

This agent loads a supervised behavior cloning model trained to imitate the
provided `ceia_baseline_agent` on player-level observations.

After training, the BC script exports weights into:

```text
baseline_bc_agent/weights/baseline_bc.pt
```
