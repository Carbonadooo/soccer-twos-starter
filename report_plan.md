# Final Project Report Plan

---

## Report Outline

### Abstract
We address the sparse-reward challenge in multi-agent 2v2 soccer (SoccerTwos) by combining
imitation learning with shaped reward fine-tuning. Training from a random policy with vanilla
PPO failed due to extreme reward sparsity. Instead, we first distill a pre-trained baseline
into a behavioral cloning (BC) policy, then fine-tune it with PPO against the baseline agent
using potential-based reward shaping. Our final agent achieves a win rate of [X]% against the
baseline, compared to [Y]% for the BC-only policy and near 0% for vanilla PPO.

---

### 1. Overview (1 paragraph)
We train agents for the SoccerTwos 2v2 environment using Ray RLLib PPO. The default sparse
reward (±1 per goal) makes it extremely difficult for an agent to learn from scratch — random
play rarely produces goals, so the gradient signal is nearly absent. We investigate three
progressive approaches: (1) vanilla PPO from random initialization, (2) behavioral cloning
(BC) from a pre-trained baseline agent, and (3) PPO fine-tuning of the BC policy with
potential-based reward shaping that encourages ball progress, defensive coverage, and support
positioning.

---

### 2. Method Description

#### Agent 1 — Vanilla PPO (Baseline)
- **Algorithm**: PPO, Ray RLLib 1.4, PyTorch
- **Environment**: `team_vs_policy`, opponent = random/still
- **Training**: 5M steps from random initialization
- **Result**: Failed to converge. With random exploration, goals occur ~once per 600-step
  episode; the sparse ±1 reward provides almost no gradient signal for learning.
- **Theoretical background**: PPO uses a clipped surrogate objective to ensure stable policy
  updates. With dense rewards PPO converges reliably; with sparse rewards the advantage
  estimates are near zero and the policy cannot improve.

#### Agent 2 — Behavioral Cloning (BC)
- **Algorithm**: Supervised learning (cross-entropy), no RL
- **Data collection**: 300 episodes of `ceia_baseline_agent` vs itself (~33k transitions)
- **Model**: 2-layer MLP (512×512), 3 classification heads for MultiDiscrete([3,3,3]) actions
- **Observation modification**: z-score normalization per dimension using statistics computed
  from the collected dataset. This maps mixed-scale features (ray fractions ~[0,1], velocities
  ~[-10,10]) to zero-mean unit-variance, stabilizing gradient flow.
- **Result**: Reached 98% action-matching accuracy against baseline play patterns.
- **Novel technique**: Imitation learning from expert demonstrations.

#### Agent 3 — BC + PPO Fine-tuning + Reward Shaping
- **Initialization**: BC weights loaded into PPO policy before any RL updates
- **Reward modification**: Potential-based shaping added on top of the sparse goal reward:
  - *Ball progress*: `0.02 × Δ(ball_x / 14)` — rewards incremental forward ball movement
  - *Shooting signal*: nearest player rewarded when ball velocity directed toward opponent goal
  - *Defensive coverage*: bonus when a player is positioned between ball and own goal
  - *Support positioning*: rewards width and depth behind ball (attack) / lane coverage (defense)
  - *Anti-crowding*: penalizes teammates within 3 units of each other
  - All components scaled by 0.15 to keep total shaped reward per episode < goal reward (±1)
- **Training**: PPO fine-tuning vs `ceia_baseline_agent`, randomly assigned blue/orange each
  episode to prevent directional bias
- **Key hyperparameters**: lr=3e-5, clip=0.1, batch=8000, γ=0.99, λ=0.95, entropy=0.01

---

### 3. Experimental Results

#### Hyperparameter Table (Agent 3 final run)

| Parameter          | Value  |
|--------------------|--------|
| Algorithm          | PPO    |
| Library            | Ray RLLib 1.4 |
| Learning Rate      | 3e-5   |
| Clip Param (ε)     | 0.1    |
| Train Batch Size   | 8000   |
| SGD Minibatch Size | 512    |
| SGD Iterations     | 15     |
| γ (discount)       | 0.99   |
| λ (GAE)            | 0.95   |
| Entropy Coeff      | 0.01   |
| VF Loss Coeff      | 0.5    |
| Hidden Layers      | [512, 512] |
| BC Training Episodes | 300  |
| BC Epochs          | 50     |

#### Win Rate Table (fill in after running evaluate)

| Agent              | vs Random Agent | vs Baseline Agent |
|--------------------|-----------------|-------------------|
| Agent 1 (PPO only) | [X] / 50        | [X] / 50          |
| Agent 2 (BC only)  | [X] / 50        | [X] / 50          |
| Agent 3 (BC+PPO+Shaping) | [X] / 50  | [X] / 50          |

#### Figures needed
1. Training curve: `episode_reward_mean` vs `timesteps_total` for Agent 1 and Agent 3
2. Overlaid comparison of Agent 1 vs Agent 3 reward curves
3. Win rate bar chart comparing all three agents

---

### 4. Analysis and Discussion

**Why vanilla PPO failed**: With sparse ±1 reward and random initialization, the agent scores
roughly 0.05 goals per episode in early training. Gradient updates carry almost no information;
the policy never escapes random behavior. This confirms that sparse rewards alone are
insufficient for soccer from scratch.

**Why BC works**: By supervising directly on 33k expert transitions, the agent bypasses the
exploration problem entirely. Starting at 98% action-matching accuracy means it already plays
at near-baseline level before any RL, confirming that imitation learning is an effective
bootstrap strategy for sparse-reward environments.

**Effect of reward shaping**: Potential-based components provide dense gradient signal every
step, guiding the agent toward tactically sound behavior. The global scale factor (×0.15)
prevents the shaped reward from overwhelming the true win/loss objective. [Discuss whether
Agent 3 outperforms Agent 2 based on actual win rates.]

**Observation normalization**: The raw 336-dim observation mixes features of different scales.
Z-score normalization reduces gradient variance and accelerates convergence, consistent with
standard RL best practice (Andrychowicz et al., 2021).

**Motivation for modifications**: We hypothesized that (1) normalizing observations would
stabilize early training by reducing the effective condition number of the input, and (2)
potential-based reward shaping would guide the policy toward goal-scoring behavior without
distorting the optimal policy (Ng et al., 1999). [Discuss whether results support or contradict
these hypotheses.]

---

### 5. References
- Schulman et al., "Proximal Policy Optimization Algorithms," arXiv 2017
- Liang et al., "RLlib: Abstractions for Distributed Reinforcement Learning," ICML 2018
- Ng et al., "Policy Invariance Under Reward Transformations: Theory and Practice," ICML 1999
- Andrychowicz et al., "What Matters in On-Policy Reinforcement Learning? A Large-Scale Empirical Study," ICLR 2021
- Unity ML-Agents Toolkit: SoccerTwos Environment

---

---

## Experiments Checklist

### Step 1: Win Rate Evaluation

Fill in `<AGENT1>`, `<AGENT2>`, `<AGENT3>` with your teammate's agent folder names.

```powershell
# Agent 1 (Vanilla PPO) vs Random
python -m soccer_twos.evaluate -m1 <AGENT1> -m2 example_player_agent -e 50

# Agent 1 (Vanilla PPO) vs Baseline
python -m soccer_twos.evaluate -m1 <AGENT1> -m2 ceia_baseline_agent -e 50

# Agent 2 (BC only) vs Random
python -m soccer_twos.evaluate -m1 <AGENT2> -m2 example_player_agent -e 50

# Agent 2 (BC only) vs Baseline
python -m soccer_twos.evaluate -m1 <AGENT2> -m2 ceia_baseline_agent -e 50

# Agent 3 (BC + PPO + Reward Shaping) vs Random
python -m soccer_twos.evaluate -m1 <AGENT3> -m2 example_player_agent -e 50

# Agent 3 (BC + PPO + Reward Shaping) vs Baseline
python -m soccer_twos.evaluate -m1 <AGENT3> -m2 ceia_baseline_agent -e 50
```

Look for `policy_win_rate` in the output for each run.

---

### Step 2: Generate Training Curves

#### For runs trained with tune.run (has progress.csv):
```powershell
python analyze_ray_results.py \
    --results-dir ray_results \
    --output-dir results_analysis \
    --x-axis timesteps_total \
    --smooth-window 10
```
Plots are saved to `results_analysis/`.

#### For specific runs only:
```powershell
python analyze_ray_results.py \
    --results-dir ray_results \
    --runs <EXPERIMENT_FOLDER_NAME> \
    --output-dir results_analysis \
    --smooth-window 10
```

#### For runs trained with manual PPOTrainer (no progress.csv):
Check if Ray wrote default logs:
```powershell
ls ~/ray_results/
```
If not present, the reward values need to be logged during a re-run or extracted from
stdout if it was saved.

---

### Step 3: Verify Each Agent Loads Correctly

Before submitting, test each agent in a fresh clone:
```powershell
python -m soccer_twos.watch -m1 <AGENT1> -m2 ceia_baseline_agent
python -m soccer_twos.watch -m1 <AGENT2> -m2 ceia_baseline_agent
python -m soccer_twos.watch -m1 <AGENT3> -m2 ceia_baseline_agent
```

---

### Step 4: Package for Submission

```powershell
# From the project root directory:
zip -r TEAMNAME_AGENT.zip TEAMNAME_AGENT/
```

Contents of the zip must include:
- `__init__.py`
- `agent.py` (or equivalent)
- Any checkpoint / model weight files
- Any helper files imported at runtime (e.g., `model.py`, normalisation stats)
