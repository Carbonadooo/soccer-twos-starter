# Team Vs Random Reward-Shaped Agent

**Agent name:** TeamVsRandomRewardShapedPPO

**Author(s):** Shaoyu

## Description

This agent loads a PPO policy trained with
`train_ray_team_vs_random_reward_shaping.py`.
The training environment uses `team_vs_policy` with both teammates jointly
controlled by one policy network.

This checkpoint is expected to be trained for the left-side team using
world-coordinate reward shaping. If you later want to deploy it on the
right-side team, add a coordinate/action conversion wrapper at evaluation time.

After training, copy the selected RLlib checkpoint file into:

```text
team_vs_random_reward_shaped_agent/checkpoint/
```

The expected checkpoint filename looks like:

```text
checkpoint-300
```

Do not copy only the `checkpoint_000300` directory name; copy the actual
`checkpoint-*` file inside it.
