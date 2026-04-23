# PPO Still Agent

**Agent name:** PPOStill

**Author(s):** Shaoyu

## Description

This agent loads a PPO policy trained with `example_ray_ppo_sp_still.py`.
The training environment uses `team_vs_policy`, `single_player=True`,
`flatten_branched=True`, and a still opponent policy.

After training, copy the selected RLlib checkpoint file into:

```text
ppo_still_agent/checkpoint/
```

The expected checkpoint filename looks like:

```text
checkpoint-100
```

Do not copy only the `checkpoint_000100` directory name; copy the actual
`checkpoint-*` file inside it.
