# KuaiRand RL Environment

RL environment for video recommendation using the KuaiRand dataset with offline IQL training.

## Quick Start

```bash
source kuairand_env/bin/activate
python demo.py
```

## Offline IQL Training

```bash
# 1. Create train/test split (80/20)
python create_train_test_split.py

# 2. Train model with reward normalization
python train_offline_improved.py

# 3. Evaluate on test set
python evaluate_improved_model.py
```

## Project Structure

```
src/
├── data_loader.py              # Dataset loader
├── environment.py              # Gymnasium environment
├── iql/                        # IQL implementation
└── training/                   # Offline training utilities

create_train_test_split.py      # Data splitting
train_offline_improved.py       # Offline IQL trainer
evaluate_improved_model.py      # Model evaluation
```

## Environment

- **State**: 64-dimensional vector (user embedding + history + context)
- **Action**: Discrete (video recommendations)
- **Reward**: 0.5 * click + 0.5 * watch_ratio
- **Episode**: 10 steps

## Dataset

KuaiRand dataset:
- 1,000 users
- 4.3M videos
- 11.7M interactions (full dataset)

Source: https://github.com/chongminggao/KuaiRand
