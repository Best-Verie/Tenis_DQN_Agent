# DQN Atari Assignment - Boxing 

## Group 4 Members:

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

experiments/irutingabo-experiments.ipynb

This section defines 10 experiment configurations arranged as 5 paired CNN/MLP groups — each pair shares identical hyperparameters so that CnnPolicy and MlpPolicy can be compared fairly under the same conditions.

results/raissa/tables/raissa_hyperparameter_results.csv
results/raissa/tables/raissa_hyperparameter_results.md

Hyperparameter Configurations
namepolicylearning_rategammabatch_sizeexploration_initial_epsexploration_final_epsexploration_fractionraissa_exp01_cnn_baselineCnnPolicy1e-40.99321.00.050.10raissa_exp02_mlp_baselineMlpPolicy1e-40.99321.00.050.10raissa_exp03_cnn_low_lrCnnPolicy5e-50.99321.00.050.10raissa_exp04_mlp_low_lrMlpPolicy5e-50.99321.00.050.10raissa_exp05_cnn_high_lrCnnPolicy2.5e-40.99321.00.050.10raissa_exp06_mlp_high_lrMlpPolicy2.5e-40.99321.00.050.10raissa_exp07_cnn_slow_epsCnnPolicy1e-40.99321.00.100.50raissa_exp08_mlp_slow_epsMlpPolicy1e-40.99321.00.100.50raissa_exp09_cnn_large_batchCnnPolicy1e-40.991281.00.050.10raissa_exp10_mlp_large_batchMlpPolicy1e-40.991281.00.050.10
Results
namepolicymean_rewardstd_rewardtrain_minutesraissa_exp10_mlp_large_batchMlpPolicy2.85.461.53raissa_exp03_cnn_low_lrCnnPolicy-1.21.175.08raissa_exp04_mlp_low_lrMlpPolicy-2.46.341.55raissa_exp09_cnn_large_batchCnnPolicy-4.06.206.34raissa_exp02_mlp_baselineMlpPolicy-6.86.851.53raissa_exp06_mlp_high_lrMlpPolicy-7.04.941.53raissa_exp05_cnn_high_lrCnnPolicy-7.28.015.04raissa_exp07_cnn_slow_epsCnnPolicy-8.09.904.82raissa_exp01_cnn_baselineCnnPolicy-9.89.415.13raissa_exp08_mlp_slow_epsMlpPolicy-15.03.741.46
Findings and Insights
Best model: raissa_exp10_mlp_large_batch with a mean reward of 2.8 — the only experiment to score positively across all 10 runs.
Surprise result — MLP beat CNN overall. The average mean reward across all 5 CNN experiments was -6.04, while the MLP average was -5.68. This is unexpected since Boxing is a pixel-based environment where CNNs are designed to have an advantage. However, at only 50,000 timesteps, the CNN likely did not have enough training time to learn useful visual features, whereas the MLP was able to update faster (roughly 1.5 min per run vs ~5 min for CNN) and squeeze more useful learning out of the same budget.
What helped — Lower learning rate. Reducing the learning rate to 5e-5 produced the best CNN result (-1.2) and the second-best MLP result (-2.4). Slower, more cautious updates appear to stabilize training at this short timestep budget, preventing the agent from bouncing around without settling on a strategy.
What helped — Larger batch size (for MLP). The MLP with batch size 128 (exp10) was the best experiment overall. Smoother gradient estimates from larger batches gave the MLP enough signal to learn a basic positive reward strategy. Interestingly, the same change hurt the CNN (exp09 scored -4.0), likely because the CNN needs more frequent updates to start learning from frames early in training.
What hurt — Higher learning rate. Both exp05_cnn_high_lr (-7.2) and exp06_mlp_high_lr (-7.0) performed worse than their respective baselines. At 2.5e-4, updates were too aggressive for the agent to stabilize, leading to inconsistent Q-value estimates and poor play.
What hurt the most — Slow epsilon decay. Extending exploration to 50% of training (exploration_fraction=0.50) was the worst decision for both policies. exp07_cnn_slow_eps scored -8.0 and exp08_mlp_slow_eps scored -15.0 (the worst result overall). Spending too much of a short 50k-step budget on random exploration left almost no time for the agent to exploit what it had learned. At this training scale, a fast epsilon decay is more practical.
CNN vs MLP conclusion: CNN is theoretically the right tool for pixel-based environments like Boxing, but it requires significantly more training time and timesteps to benefit from its visual feature extraction. With only 50k timesteps and ~5 minutes of training, the CNN did not have enough budget to outperform MLP. A longer run (200k+ timesteps) would likely reverse the results in CNN's favor.
Required Artifacts

Best model (assignment name): results/raissa/models/dqn_model.zip
Best model copy: results/raissa/models/best_dqn_boxing.zip
Hyperparameter table CSV: results/raissa/tables/raissa_hyperparameter_results.csv
Hyperparameter table Markdown: results/raissa/tables/raissa_hyperparameter_results.md
Gameplay video (optional from notebook cell): results/raissa/videos/playback/*.mp4

Colab + Google Drive Export
The notebook includes export logic that copies key artifacts to:

/content/drive/MyDrive/Boxing_dqn_agent/raissa
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
