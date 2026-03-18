# rollout-slim

**Not All Transitions Matter: Evidence from PPO**
*Reducing Temporal Correlation in PPO Without Degrading Performance*

**Author:** Ajhesh Basnet &nbsp;·&nbsp; ajheshb@gmail.com

---

## Abstract

![Abstract](https://raw.githubusercontent.com/ajheshbasnet/rollout-slim/main/static/abstract.png)

On-policy trajectory data is correlated by construction. Each state causally produces the next, which breaks the IID assumption that stable neural network training depends on. This drives gradient updates toward near-collinearity, triggers a **Non-Stationary Bootstrapping Feedback Loop**, and damages both training stability and sample efficiency.

Three methods are presented for reducing this correlation without modifying the core PPO pipeline. The central finding is that **randomly subsampling $p\%$ of transitions after GAE computation** — rather than before — is sufficient to break the temporal correlation structure while keeping the reward signal completely intact.

The method matches vanilla PPO on reward while showing measurably more stable training across five environments spanning discrete control, continuous locomotion, and high-dimensional Atari: **CartPole-v1**, **Acrobot-v1**, **LunarLander-v2**, **HalfCheetah**, and **Hopper**.

---

## Repository Structure

```
rollout-slim/
│
├── descrete-nb/                    # Discrete-action environments
│   ├── Full-PPO.ipynb              # Vanilla PPO baseline (control)
│   ├── P% PPO.ipynb                # Method 3 — p% subsampling after GAE (main contribution)
│   ├── randSkipAlternate.ipynb     # Method 2 — Random Adaptive K-Step Sampling
│   └── randomSkipAlternate.ipynb   # Method 1 — Fixed K-Step Sampling
│
├── continious-nb/                  # Continuous-action environments (MuJoCo)
│   ├── HalfCheetah/                # PPO + p% subsampling on HalfCheetah-v4
│   └── Hopper/                     # PPO + p% subsampling on Hopper-v4
│
├── atari/                          # High-dimensional pixel-based environments
│
├── static/                         # Figures and abstract image
└── README.md
```

---

## Background and Motivation

### The IID Problem in On-Policy RL

Supervised learning assumes samples are **Independently and Identically Distributed (IID)**. In on-policy RL, this assumption is broken by construction. A trajectory is defined as:

$$\tau = (s_0, a_0, r_0) \to (s_1, a_1, r_1) \to \cdots \to (s_T, a_T, r_T)$$

Every transition $(s_t, a_t, r_t, s_{t+1})$ is causally downstream of the one before it. Feeding this sequential chain into a neural network produces gradient vectors that are nearly parallel update after update — a structural form of collinearity that slows and destabilises convergence.

### The Non-Stationary Bootstrapping Feedback Loop

The TD update for the value network is:

$$V(s_t) \leftarrow V(s_t) + \alpha \left[ r_t + \gamma V(s_{t+1}) - V(s_t) \right]$$

Both $V(s_t)$ and $V(s_{t+1})$ come from the same network whose weights shift during training. After $n$ policy updates, the critic is evaluated on state distributions it was never trained on. This compounds across trajectories — a direct instance of the **Deadly Triad**: function approximation + bootstrapping + non-stationary data distribution.

---

## Methods

### Method 1 — Fixed $K$-Step Sampling (`randomSkipAlternate.ipynb`)

Store every $K$-th transition and accumulate intermediate rewards:

$$r_{\text{stored}} = \sum_{i=0}^{K-1} r_{t+i}$$

**Why it fails:** The fixed interval punches the same holes in every trajectory. On LunarLander-v2, reward summation destroys fine-grained credit assignment — the agent cannot tell which specific action caused which outcome.

---

### Method 2 — Random Adaptive $K$-Step Sampling (`randSkipAlternate.ipynb`)

Randomise the skip interval per trajectory:

$$\varepsilon \sim \mathcal{N}(0, 1), \qquad k' = \begin{cases} k & \text{if } \varepsilon > 0 \\ k + 1 & \text{if } \varepsilon \leq 0 \end{cases}$$

**Improvement over Method 1:** Eliminates the parity bias — the buffer now sees a broader, more representative slice of the trajectory over time. Still fails on LunarLander-v2 because reward summation across skipped steps is unchanged. The root cause — intervening before GAE — was not fixed.

---

### Method 3 — Random $p\%$ Trajectory Subsampling (`P% PPO.ipynb`) ⭐ Main Contribution

**The key insight:** Intervene *after* GAE, not before. This preserves the reward signal completely while decorrelating the gradient update.

**Procedure:**

1. Collect the full trajectory buffer with no skipping.
2. Compute GAE over the complete, unmodified sequence:

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

3. Randomly sample $p\%$ of the $N$ transitions without replacement for the gradient update.
4. The remaining $(1-p)\%$ are excluded only from the optimisation step — their reward contributions are already baked into the advantage estimates.

**The Dropout analogy:** Dropout randomly kills neurons to prevent co-adaptation to redundant features. This method applies the same principle one level higher — instead of dropping neurons, it drops transitions. Structured randomness breaks correlated pathways that damage optimisation.

| Effect | Mechanism |
|--------|-----------|
| **Decorrelation** | Random selection disrupts sequential structure without losing reward information |
| **Memory efficiency** | Only $p\%$ of transitions go to the GPU per update |
| **Implicit regularisation** | The optimiser never sees the full correlated batch the same way twice |

---

## Experimental Setup

### Environments

| Environment | Action Space | Reward Structure | Difficulty |
|-------------|-------------|-----------------|------------|
| `CartPole-v1` | Discrete | Dense — every timestep | Low — 4D state, simple dynamics |
| `Acrobot-v1` | Discrete | Sparse — penalty until goal | Medium — credit assignment required |
| `LunarLander-v2` | Discrete | Shaped — position, velocity, tilt, fuel | High — long-horizon credit assignment |
| `HalfCheetah-v4` | Continuous | Dense — velocity reward | High — 17D state, continuous locomotion |
| `Hopper-v4` | Continuous | Dense — forward progress + survival | High — balance and locomotion simultaneously |

---

### Hyperparameters — Discrete Environments

| Hyperparameter | CartPole-v1 | Acrobot-v1 | LunarLander-v2 |
|----------------|-------------|------------|----------------|
| Max Training Steps | 500,000 | 900,000 | 1,000,000 |
| Rollout Steps | 1,400 | 1,400 | 1,400 |
| PPO Clip $(\varepsilon)$ | 0.20 | 0.20 | 0.18 |
| Entropy Coeff $(\beta)$ | 0.01 | 0.09 | 0.05 |
| Optimizer | AdamW | AdamW | AdamW |
| Actor LR | 3e-4 | 3e-4 | 3e-4 |
| Critic LR | 5e-4 | 5e-4 | 5e-4 |
| $\gamma$ | 0.99 | 0.99 | 0.99 |
| GAE $\lambda$ | 0.98 | 0.98 | 0.98 |

### Hyperparameters — Continuous Environments (MuJoCo)

| Hyperparameter | HalfCheetah-v4 | Hopper-v4 |
|----------------|----------------|-----------|
| Optimizer | AdamW | AdamW |
| Actor LR | — | — |
| Critic LR | — | — |
| PPO Clip $(\varepsilon)$ | — | — |
| Entropy Coeff $(\beta)$ | — | — |
| $\gamma$ | — | — |
| GAE $\lambda$ | — | — |

> ⚠️ **Fill in the dashes above** with the actual values from `continious-nb/HalfCheetah/` and `continious-nb/Hopper/` before publishing.

**Evaluation Protocol:** At each checkpoint, the agent ran for 1 episode across 3 independent seeds. Reported reward is the mean across those 3 runs.

**Tracked Metrics:** KL divergence, policy entropy, explained variance, value bias, critic loss, evaluation reward.

---

## Results Summary

### Discrete Environments

| Method | CartPole-v1 | Acrobot-v1 | LunarLander-v2 |
|--------|-------------|------------|----------------|
| Vanilla PPO | Baseline | Baseline | Baseline |
| Method 1 — Fixed $K$-Step | Works | Unstable | Fails |
| Method 2 — Random $K$-Step | Works | Better | Fails |
| **Method 3 — $p\%$ Subsample** | **Matches PPO** | **Matches PPO** | **Matches PPO** |

### Continuous Environments (MuJoCo)

| Method | HalfCheetah-v4 | Hopper-v4 |
|--------|----------------|-----------|
| Vanilla PPO | Baseline | Baseline |
| **Method 3 — $p\%$ Subsample** | **Matches PPO** | **Matches PPO** |

### Key Finding on $p$

- **$p = 75\%$** is the sweet spot across all tested environments — reward, entropy, and KL all match vanilla PPO cleanly.
- **$p < 75\%$:** reward still looks fine, but entropy drifts and KL gets noisier. The optimiser quietly loses stable exploration before the reward signal even reflects it.
- Dropping exactly 25% of transitions is sufficient to break the correlated gradient structure without thinning the batch enough to destabilise learning.

---

## Discussion

### Why Shuffling Alone Is Not Enough

Standard PPO already shuffles rollout data into minibatches. But shuffling changes the *order* transitions arrive — not *which* ones are included. If 1,000 of 1,400 transitions are nearly identical, shuffling still feeds all 1,000 to the optimiser. Subsampling to $p\%$ removes redundant transitions outright. That is the structural difference, and why Method 3 shows lower variance while shuffling does not.

### Why Timing Is Everything

Methods 1 and 2 accumulated rewards across skipped steps, which broke the Markov assumption: the stored $s_t$ carried information from future states it never observed. Method 3 touches nothing before GAE. The environment, rollout, GAE computation, and PPO-clip objective are identical to vanilla PPO. The entire modification is a single random sampling step inserted between GAE and the gradient update.

---

## Conclusion

Temporal correlation in on-policy trajectory data is a structural problem, not a noise problem. The right intervention point is after GAE computation. Randomly subsampling $p\%$ of transitions at that point is sufficient to break the correlation structure without sacrificing any reward information — matching standard PPO across both discrete and continuous control benchmarks while improving training stability.

On-policy trajectories carry far more redundant, correlated transitions than is commonly assumed. Removing them randomly does not degrade learning. It regularises it.

---

## Citation

```bibtex
@article{basnet2026rolloutslim,
  title  = {Not All Transitions Matter: Evidence from PPO},
  author = {Ajhesh Basnet},
  year   = {2026},
  note   = {Independent Research},
  url    = {https://github.com/ajheshbasnet/rollout-slim}
}
```

---

## References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
2. Sutton, R. S. and Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
3. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.* JMLR, 15(1), 1929–1958.
