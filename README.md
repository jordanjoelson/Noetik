# Noetik: Interpretable RL for Video Recommendations

RL environment for video recommendation using the KuaiRand dataset with offline IQL training and gradient-based interpretability.

## Dataset

KuaiRand dataset:
- 1,000 users
- 4.3M videos
- 11.7M interactions (full dataset)

Source: https://github.com/chongminggao/KuaiRand
# Goals
- Use social media interaction data to understand how people engage with content and what keeps them returning.

- Analyze user behavior to uncover patterns of attention, preference, and emotional response in digital environments.

- Model engagement as a reinforcement process to explore how social media platforms shape user habits.

- Build a framework that can extend to other domains, such as psychometric or behavioral datasets, for broader studies of human decision-making.

# Model Overview & Architecture
## Overview
- Environment: Based on the KuaiRand dataset, representing realistic user–video interactions on social media platforms.

- Agent: A reinforcement learning agent trained to model and explain user engagement behavior.

- Policy Network: Transformer-based architecture with attention to capture how past interactions influence current content choices.

- Interpretability Layer: Translates model reasoning into human-understandable explanations showing why certain actions or recommendations occur.

- Trainer: Uses a multi-objective optimization that balances engagement modeling with interpretability and transparency.

## Architecture
 <img width="815" height="378" alt="Model Architecture" src=assets/noetik-architecture.JPG/>
 
# Novelty
- Treats social media interaction as a reinforcement learning process, simulating how users learn and adapt through feedback.

- Embeds interpretability directly into the RL design, allowing insight into both model and human decision processes.

- Uses attention not just for prediction accuracy but as a tool for behavioral interpretation.

- Shifts the focus from maximizing engagement to understanding the cognitive and emotional mechanisms behind it.

- Provides a foundation that can generalize to psychometric and behavioral research beyond social media.

# Current Dataset:
KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos
  - https://github.com/chongminggao/KuaiRand

The following figure gives an example of the dataset. It illustrates a user interaction sequence along with the user's rich feedback signals.

<img width="815" height="378" alt="KuaiRand-homepage" src="https://github.com/user-attachments/assets/13cde850-635a-48b7-9f32-9ff5bad7946e" />

These feedback signals are collected from the two main user interfaces (UI) in the Kuaishou APP shown as follows:

<img width="815" height="1077" alt="kuaishou-app" src="https://github.com/user-attachments/assets/bb91f66a-f448-49af-b9ff-b4d4aa9a01d9" />


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
interpret.py                     # Gradient-based interpretability
```

## Environment

- **State**: 128-dimensional vector (user embedding + history + context)
- **Action**: Discrete (video recommendations)
- **Reward**: 0.5 * click + 0.5 * watch_ratio
- **Episode**: 10 steps
