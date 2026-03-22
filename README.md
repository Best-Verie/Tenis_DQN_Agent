# DQN Atari Assignment - Boxing 

## Group 3 Members:

- Elyse Marie Uyiringiye
- Nice Eva Karabaranga
- Best Verie Iradukunda
- Raissa Irutingabo

## Model Files/Videos 

[https://drive.google.com/drive/folders/18KEPQ7AlXoZxDiDu8lFAwod63G7g-8rQ?usp=sharing]
## Best Model Playing: 

[Best Model](https://drive.google.com/file/d/1rm9EKlDZ269LD7_plWfo2yjR2vQQzB_5/view?usp=drive_link)

## Overview

This project implements a Deep Q-Network (DQN) agent trained to play the Atari Boxing-v5 environment using Stable Baselines3 and Gymnasium. This requires training and comparing multiple DQN configurations with different hyperparameters and policy architectures (CNNPolicy vs MLPPolicy).


## Environment
- Gymnasium Atari environment: `ALE/Boxing-v5`
- Framework: Stable-Baselines3 `DQN`

## Scripts
- `scripts/train.py`: trains different DQN experiments  and saves model artifacts
- `scripts/play.py`: loads trained best model and runs evaluation gameplay


## Key Libraries

stable-baselines3 (≥2.3.2): DQN algorithm and policy implementations
gymnasium[atari] (≥1.0.0): Atari environment wrapper
ale-py (≥0.10.1): Atari Learning Environment interface
torch: Neural network backend
pandas, matplotlib: Data analysis and visualization

## Installation

# Install dependencies
`pip install -r requirements.txt`

# On some systems, may need to install ROMs separately
`AutoROM --accept-license`

# Group Members Contribution

## Best Verie Experiments

# Boxing DQN Experiments (ALE/Boxing-v5)


## Best Verie's Experiment Summary (11 experiments)

| Experiment | Policy | Mean Reward | Std Reward | Train Time (min) | Notes |
|-----------|--------|------------|------------|------------------|------|
| boxing_exp01_baseline_cnn | CnnPolicy | **5.9** | 5.50 | 14.23 |  Best overall |
| boxing_exp05_high_gamma_cnn | CnnPolicy | 4.7 | 3.66 | 14.11 | Higher gamma |
| boxing_exp08_less_exploration_cnn | CnnPolicy | 4.1 | 4.50 | 14.36 | Less exploration |
| boxing_exp02_small_batch_cnn | CnnPolicy | 3.8 | 5.27 | 13.62 | Smaller batch |
| boxing_exp04_low_gamma_cnn | CnnPolicy | 1.2 | 3.91 | 14.08 | Lower gamma |
| boxing_exp06_gamma_zero_cnn | CnnPolicy | -0.3 | 6.33 | 14.16 | No future reward |
| boxing_exp03_large_batch_cnn | CnnPolicy | -0.6 | 3.29 | 14.75 | Large batch |
| boxing_exp07_more_exploration_cnn | CnnPolicy | -0.7 | 5.08 | 13.96 | More exploration |
| boxing_exp09_more_updates_cnn | CnnPolicy | -1.3 | 4.27 | 19.49 | More gradient steps |
| boxing_exp11_small_batch_mlp | MlpPolicy | -14.4 | 5.64 | 7.04 | MLP policy |
| boxing_exp10_baseline_mlp | MlpPolicy | -27.3 | 4.47 | 7.18 | Worst performance |


##  Best Model

The best-performing model is:boxing_exp01_baseline_cnn with mean reward of 5.9
Files:
## Key files

experiments/BestVerie_experiments.ipynb: Full training pipeline with callbacks and visualization
Hyperparameter_tables/BestVerie_hyperparameter_results.csv: Summary results table
results/BestVerie.zip: logs,models, everything related to the training process
Video Demonstration:

## Demo

[![Watch my play demo](https://via.placeholder.com/800x400?text=Watch+Boxing+AI+Demo)](!https://drive.google.com/file/d/14716OGsd0Bl2DVU9waerE_U_6jVZnpA7/view?usp=sharing)



## Raissa Experiments (10 Configurations)
The notebook used is:
- `experiments/raissa_experiments.ipynb`

It now defines 10 unique experiment combinations including both policy types:
- 8 runs with `CnnPolicy`
- 2 runs with `MlpPolicy`

## Hyperparameter Results Table
After running the notebook, use:
- `results/raissa/tables/raissa_hyperparameter_results.csv`
- `results/raissa/tables/raissa_hyperparameter_results.md`

Hyperparameter table - Raissa:

| Member Name | Hyperparameter Set | Noted Behavior |
|-------------|-------------------|----------------|
| Raissa | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 [CnnPolicy] | [Exp 1 - CNN Baseline] Standard DQN with CNN. Reference point for all CNN experiments. Agent struggles at 50k steps — Tennis requires more training to show meaningful rewards. Mean Reward: -9.8 ± 9.41 |
| Raissa | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 [MlpPolicy] | [Exp 2 - MLP Baseline] Same hyperparameters as Exp 1, only policy differs. MLP cannot extract spatial features from pixels. Confirms CNN vs MLP gap. Mean Reward: -6.8 ± 6.85 |
| Raissa | lr=5e-5, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 [CnnPolicy] | [Exp 3 - CNN Low LR] Lower learning rate slows convergence significantly. Agent learns more cautiously but reward remains negative at 50k steps. Mean Reward: -1.2 ± 1.17 |
| Raissa | lr=5e-5, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 [MlpPolicy] | [Exp 4 - MLP Low LR] MLP with reduced learning rate. Slower convergence compounds the spatial feature limitation of MLP. Mean Reward: -2.4 ± 6.34 |
| Raissa | lr=2.5e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 [CnnPolicy] | [Exp 5 - CNN High LR] Higher learning rate causes Q-value instability. Loss spikes during training. High reward variance indicates unstable policy. Mean Reward: -7.2 ± 8.01 |
| Raissa | lr=2.5e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 [MlpPolicy] | [Exp 6 - MLP High LR] MLP with high learning rate. Aggressive updates on a policy without spatial reasoning leads to poor performance. Mean Reward: -7.0 ± 4.94 |
| Raissa | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.10, epsilon_decay=0.50 [CnnPolicy] | [Exp 7 - CNN Slow Epsilon Decay] Exploration maintained for 50% of training. Diverse replay buffer but insufficient steps to benefit fully at 50k. High variance. Mean Reward: -8.0 ± 9.90 |
| Raissa | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.10, epsilon_decay=0.50 [MlpPolicy] | [Exp 8 - MLP Slow Epsilon Decay] Extended exploration with MLP policy. Worst experiment — slow decay combined with no spatial reasoning produces the lowest reward. Mean Reward: -15.0 ± 3.74 |
| Raissa | lr=1e-4, gamma=0.99, batch=128, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 [CnnPolicy] | [Exp 9 - CNN Large Batch] Larger batch produces smoother but slower gradient updates. Fewer updates per timestep limits learning at 50k steps. Mean Reward: -4.0 ± 6.20 |
| Raissa | lr=1e-4, gamma=0.99, batch=128, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 [MlpPolicy] | [Exp 10 - MLP Large Batch] **Best experiment.** MLP with large batch — smoother updates reduce the noise inherent in MLP training. Surprisingly outperforms all CNN experiments at 50k steps. Mean Reward: 2.8 ± 5.46 |

## Noted Behavior / Insights
Fill after runs:
- Which hyperparameter changes improved performance?
- Which settings harmed performance?
- Why the final best configuration performed best?
- CNN vs MLP comparison result for Tennis.

## Required Artifacts
- Best model (assignment name): `results/raissa/models/dqn_model.zip`
- Best model copy: `results/raissa/models/best_dqn_tennis.zip`
- Hyperparameter table CSV: `results/raissa/tables/raissa_hyperparameter_results.csv`
- Hyperparameter table Markdown: `results/raissa/tables/raissa_hyperparameter_results.md`
- Gameplay video (optional from notebook cell): `results/raissa/videos/playback/*.mp4`

## Colab + Google Drive Export
The notebook includes export logic that copies key artifacts to:
- `/content/drive/MyDrive/Boxing_dqn_agent/raissa`

So you can access the model and results table even after Colab runtime disconnects.

## MarieAElyse — ALE/Boxing-v5

### Environment
| Property | Value |
|----------|-------|
| Environment | `ALE/Boxing-v5` |
| Action Space | `Discrete(18)` |
| Observation Space | `Box(0, 255, (210, 160, 3), uint8)` |
| Algorithm | DQN + CnnPolicy |
| Total Timesteps | 100,000 per experiment |

---

### Hyperparameter Table

| Member | Hyperparameter Set | Noted Behavior |
|--------|-------------------|----------------|
| Elyse | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 | [Exp 1 - Baseline] Stable training. Reward improves steadily. Agent learns to land punches over time. Reference point for all other experiments. Mean Reward: 3.8 |
| Elyse | lr=5e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 | [Exp 2 - High LR] Q-values diverge catastrophically. Training loss spikes. lr=5e-4 is too aggressive — worst experiment. Mean Reward: -41.0 |
| Elyse | lr=1e-5, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 | [Exp 3 - Low LR] Very slow convergence. Agent still near-random at midpoint. Rewards far below baseline. Mean Reward: -4.4 |
| Elyse | lr=1e-4, gamma=0.90, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 | [Exp 4 - Low Gamma] Agent is myopic — ignores long-term scoring. Rewards plateau lower than baseline. Mean Reward: 1.4 |
| Elyse | lr=1e-4, gamma=0.999, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 | [Exp 5 - High Gamma] Agent values long-term strategy. Slightly slower early learning but more deliberate play. Mean Reward: 3.4 |
| Elyse | lr=1e-4, gamma=0.99, batch=128, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 | [Exp 6 - Large Batch] Smoother loss curve but fewer updates per timestep. Final performance below baseline. Mean Reward: -1.2 |
| Elyse | lr=1e-4, gamma=0.99, batch=16, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 | [Exp 7 - Small Batch] Noisy gradients but frequent updates. Surprisingly good performance. Mean Reward: 7.4 |
| Elyse | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.10, epsilon_decay=0.50 | [Exp 8 - Slow Epsilon Decay] Extended exploration fills replay buffer with diverse transitions. Mean Reward: 3.4 |
| Elyse | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.05 | [Exp 9 - Fast Epsilon Decay] Commits to exploitation early. Best result — Boxing is simple enough that fast exploitation beats extended exploration. **Mean Reward: 11.2 ✅ Best** |
| Elyse | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 [MlpPolicy] | [Exp 10 - MLP Ablation] Same hyperparameters as Exp 1, only policy differs. Cannot extract spatial features. 37-point gap vs CNN confirms CnnPolicy is essential. Mean Reward: -33.2 |

---

### Best Model — exp09_fast_eps

| Metric | Value |
|--------|-------|
| Mean Reward (training eval) | 11.2 ± 6.76 |
| Mean Reward (live play) | 8.6 ± 1.85 |
| Win Rate | 5W / 0D / 0L (100%) |
| Best Episode | 11.0 |
| Worst Episode | 6.0 |

---

### Key Insights

- **Best config:** `exp09_fast_eps` — fast epsilon decay (5% of steps) worked best for Boxing because the environment is simple enough to benefit from early exploitation
- **Worst config:** `exp02_high_lr` — lr=5e-4 caused Q-value divergence, scoring -41.0
- **CNN vs MLP:** CnnPolicy scored 3.8 vs MlpPolicy -33.2 with identical hyperparameters — a 37-point gap proving CNN is essential for pixel-based Atari
- **Surprise finding:** Fast epsilon decay outperformed slow decay, contrary to theory — Boxing's dense reward signal allows the agent to learn a good policy quickly

---

### Gameplay Video

>  [Watch the agent play Boxing](videos/boxing_gameplay.avi)

Agent uses **GreedyQPolicy** (`exploration_rate=0.0` → always picks `argmax Q(s,a)`).

## Nice Eva Karabaranga - ALE/Boxing-v5

### Environment
| Property | Value |
|----------|-------|
| Environment | `ALE/Boxing-v5` |
| Action Space | `Discrete(18)` |
| Observation Space | `Box(0, 255, (210, 160, 3), uint8)` |
| Algorithm | DQN + CnnPolicy |
| Total Timesteps | 100,000 per experiment |

---

### Hyperparameter Table

| Member | Hyperparameter Set | Noted Behavior |
|--------|-------------------|----------------|
| Nice | lr=2.5e-4, gamma=0.95, batch=64, eps_start=1.0, eps_end=0.02, eps_fraction=0.20 | [Exp 1 - Baseline] Moderate config, stable reference point. Mean Reward: -0.20 |
| Nice | lr=2.5e-4, gamma=0.95, batch=64, eps_start=1.0, eps_end=0.0, eps_fraction=0.20 | [Exp 2 - Zero Eps End] Fully greedy at end, there's no residual exploration. **Best performer. Mean Reward: +4.40 ✅ Best** |
| Nice | lr=2.5e-4, gamma=0.0, batch=64, eps_start=1.0, eps_end=0.02, eps_fraction=0.20 | [Exp 3 - Zero Gamma] Fully myopic agent ignores all future rewards. Expected poor performance confirmed. Mean Reward: -2.80 |
| Nice | lr=1e-3, gamma=0.95, batch=64, eps_start=1.0, eps_end=0.02, eps_fraction=0.20 | [Exp 4 - Very High LR] Aggressive updates cause unstable Q-values. Worst CNN experiment. Mean Reward: -24.80 |
| Nice | lr=1e-6, gamma=0.95, batch=64, eps_start=1.0, eps_end=0.02, eps_fraction=0.20 | [Exp 5 - Tiny LR] Near-zero learning rate: agent barely updates. Stagnant performance. Mean Reward: -11.80 |
| Nice | lr=2.5e-4, gamma=0.999, batch=256, eps_start=1.0, eps_end=0.02, eps_fraction=0.20 | [Exp 6 - Large Batch + High Gamma] Stable gradients with strong future valuation. Mean Reward: +0.40 |
| Nice | lr=2.5e-4, gamma=0.95, batch=64, eps_start=1.0, eps_end=0.02, eps_fraction=0.01 | [Exp 7 - Instant Exploit] Epsilon collapses in the first 1% of training; almost no exploration phase. Mean Reward: +2.00 |
| Nice | lr=2.5e-4, gamma=0.95, batch=64, eps_start=1.0, eps_end=0.02, eps_fraction=0.90 | [Exp 8 - Full Explore] Explores for 90% of training: agent learns slowly and struggles to exploit. Mean Reward: 0.00 |
| Nice | lr=2.5e-4, gamma=0.99, batch=64, eps_start=1.0, eps_end=0.01, eps_fraction=0.15 | [Exp 9 - Best Guess] Balanced config combining good gamma and low eps_end. Mean Reward: +1.00 |
| Nice | lr=2.5e-4, gamma=0.99, batch=64, eps_start=1.0, eps_end=0.0, eps_fraction=0.20 | [Exp 11 - Zero Eps + High Gamma] Greedy convergence combined with strong future valuation. Mean Reward: -0.20 |
| Nice | lr=2.5e-4, gamma=0.95, batch=128, eps_start=1.0, eps_end=0.0, eps_fraction=0.20 | [Exp 12 - Zero Eps + Large Batch] Larger batch stabilises gradients with fully greedy end. Mean Reward: -0.40 |
| Nice | lr=5e-4, gamma=0.99, batch=64, eps_start=1.0, eps_end=0.0, eps_fraction=0.15 | [Exp 13 - Tuned LR + Zero Eps] Slightly higher LR with zero eps and high gamma; LR too high for zero eps. Mean Reward: -20.20 |

---

### Best Model: exp02_zero_eps_end

| Metric | Value |
|--------|-------|
| Mean Reward | 4.40 ± 3.72 |
| Policy | CnnPolicy |
| Key Setting | eps_end=0.0 (fully greedy convergence) |

---

### Key Insights

- **Best config:** `exp02_zero_eps_end` :setting epsilon_end to 0.0 forces fully greedy exploitation at the end of training, giving the highest reward
- **Worst config:** `exp04_very_high_lr` : lr=1e-3 caused Q-value instability, scoring -24.80
- **Zero gamma finding:** Setting gamma=0.0 makes the agent fully myopic, meaning it only optimises for immediate reward and cannot learn long-term boxing strategy
- **Exploration tradeoff:** Both instant exploitation (exp07) and full exploration (exp08) underperformed, confirming that a balanced epsilon decay is important

---

### Artifacts
- Notebook: `experiments/Nice_experiments.ipynb`
- Best model: `Nice_dqn_model.zip`
- Hyperparameter table: `hyperparameter_table_nice.csv`
- Reward comparison: `assets/nice_reward_comparison.png`
- Training curves: `assets/nice_training_curves.png`

## Demo: 
[My best model](https://drive.google.com/file/d/1YmAl1JB5iatSVjZoA1seLZj41rOzOwoL/view?usp=drive_link) 
