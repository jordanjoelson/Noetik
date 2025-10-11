# KuaiRand RL Environment

RL environment for video recommendation using the KuaiRand dataset.

## Quick Start

```bash
source kuairand_env/bin/activate
python demo.py
```

## Project Structure

```
src/
├── data_loader.py    # Dataset loader
├── environment.py    # Gymnasium environment
└── agent.py          # Agent implementation

demo.py              # Demo script
test_env.py          # Integration tests
```

## Environment

- **State**: 64-dimensional vector
- **Action**: Discrete (7,388 videos)
- **Reward**: Composite score (click + watch ratio + like)
- **Episode**: 10 steps

## Dataset

KuaiRand-1K contains:
- 1,000 users
- 7,388 videos
- 43,026 interactions

Source: https://github.com/chongminggao/KuaiRand

## Testing

Run validation tests:
```bash
python test_env.py
```
