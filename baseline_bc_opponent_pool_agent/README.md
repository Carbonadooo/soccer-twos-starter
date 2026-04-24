Packaged agent for PPO finetuning with sampled opponents.

Drop the chosen RLlib `checkpoint-*` file into `checkpoint/` before evaluation:

- `python -m soccer_twos.watch -m baseline_bc_opponent_pool_agent`

The loader automatically adapts to different hidden sizes found in the checkpoint.
