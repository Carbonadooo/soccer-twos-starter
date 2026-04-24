# Team Vs Random Agent

**Agent name:** TeamVsRandomPPO

**Author(s):** Shaoyu

## Description

This agent loads a PPO policy trained with `train_ray_team_vs_random.py`.
The training environment uses `team_vs_policy` with both teammates jointly
controlled by one policy network.

After training, copy the selected RLlib checkpoint file into:

```text
team_vs_random_agent/checkpoint/
```

The expected checkpoint filename looks like:

```text
checkpoint-300
```

Do not copy only the `checkpoint_000300` directory name; copy the actual
`checkpoint-*` file inside it.
