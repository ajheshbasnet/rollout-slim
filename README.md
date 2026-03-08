# rollout-slim

Investigating temporal correlation in PPO rollouts — dropping 25% of transitions after GAE still matches full PPO performance, and sometimes beats it.

---

## Overview

In on-policy reinforcement learning, trajectory data is inherently temporally correlated. Consecutive states are nearly identical because each state is a direct causal consequence of the previous one. This means a standard PPO rollout of 1400 steps does not contain 1400 independent training samples — it contains a highly redundant, causally chained sequence where a significant fraction of transitions carry near-duplicate gradient signal.

This repository investigates three methods to reduce that temporal correlation in PPO rollouts without degrading training performance. The key finding is that randomly subsampling 75% of collected transitions **after** GAE computation — and training only on that subset — achieves reward curves nearly identical to vanilla PPO, with measurably more stable training metrics across all experiments.

---

## Experiments

| File | Description |
|---|---|
| `normal_ppo.ipynb` | Vanilla PPO baseline — full rollout, no subsampling |
| `p%-ppo.ipynb` | Method 3: Random p% subsampling after GAE computation |
| `skip-k-step.ipynb` | Method 1: Fixed K-step sampling with reward accumulation |
| `randSkipAlternate.ipynb` | Method 2: Random adaptive K-step sampling |

---

## Key Result

Even after dropping 25% of transitions from 1400 rollout steps per update, the following metrics remained nearly identical to vanilla PPO — and in several cases were more stable:

- KL divergence
- Policy entropy
- Explained variance
- Value bias
- Critic loss
- Evaluation reward (averaged over 3 runs per checkpoint)

This suggests that a substantial fraction of transitions in a standard on-policy trajectory are redundant due to temporal correlation, and that randomly removing them does not degrade learning — it regularizes it.

---

## How to Reproduce

To verify the core finding yourself, run `normal_ppo.ipynb` and `p%-ppo.ipynb` with the same hyperparameters listed below. You should observe very similar reward curves, with `p%-ppo` showing slightly more stable training metrics.

---

## Hyperparameters

The following configuration was used across all experiments.

| Hyperparameter | CartPole-v1 | LunarLander-v2 |
|---|---|---|
| Max Training Steps | 500,000 | 1,000,000 |
| Rollout Steps per Update | 1400 | 1400 |
| PPO Clip (epsilon) | 0.2 | 0.18 |
| Entropy Coefficient (beta) | 0.01 | 0.05 |
| Actor Network Parameters | 17,026 | 17,412 |
| Critic Network Parameters | 16,961 | 17,217 |
| Gamma | 0.99 | 0.99 |
| GAE Lambda | 0.98 | 0.98 |
| Subsampling Fraction (p) | 75% | 75% |

---

## Environments

- **CartPole-v1** — Low-dimensional discrete control. Methods 1 and 2 showed partial success here due to the simple, dense reward signal.
- **LunarLander-v2** — Moderate complexity with a shaped reward signal. Methods 1 and 2 failed here due to loss of temporal credit assignment. Method 3 matched vanilla PPO performance.

---

## Why It Works

Standard PPO already shuffles its minibatches before each gradient update. However, shuffling reorders transitions — it does not remove the underlying redundancy. Having 1,000 nearly identical transitions in a shuffled batch still biases the gradient in the same direction, because the bias comes from the statistical similarity of the samples, not their ordering.

Randomly subsampling down to 750 transitions removes a fraction of those redundant samples entirely, reducing the effective collinearity of the gradient signal regardless of order. The intuition is similar to Dropout in neural networks: just as randomly deactivating neurons breaks co-adaptation between redundant feature detectors, randomly deactivating correlated transitions breaks co-adaptation between redundant gradient directions.

Crucially, the subsampling happens **after** GAE computation — so the full reward signal is always used for advantage estimation. Only the gradient update step sees the reduced, decorrelated subset.

---

## Related Paper

This repository accompanies the paper:

**Not All Transitions Matter: Evidence from PPO**  
Ajhesh Basnet — ajheshb@gmail.com

---

## License

MIT
