import torch
import torch.nn as nn
import numpy as np
from interpret import GradientDescentInterpreter 

class DummyQNetwork(nn.Module):
    def __init__(self, state_dim=128, action_dim=1):
        super().__init__()
        # simple 2-layer net that outputs scalar Q-value
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state, action):
        # concatenate state + action along last dim
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class FakeAgent:
    def __init__(self, state_dim=128, action_dim=1):
        self.q1 = DummyQNetwork(state_dim, action_dim)


def test_interpretability():
    agent = FakeAgent()
    interpreter = GradientDescentInterpreter(agent)

    np.random.seed(42)
    state = np.random.rand(128).astype(np.float32)
    action = np.random.uniform(-1, 1)

    print("testing now")
    print(f"Simulated state shape: {state.shape}, action: {action:.3f}\n")

    importance = interpreter.compute_importance(state, action)
    print("Gradient-based importance per feature group:")
    for k, v in importance.items():
        print(f"  {k:20s}: {v:.6f}")

    top_features = interpreter.get_top_influential_features(state, action, top_k=3)
    print("\nTop influential features:")
    for name, val in top_features:
        print(f"  {name:20s}: {val:.6f}")

    explanation = interpreter.generate_gradient_explanation(state, action)
    print("\nGenerated explanation:")
    print(" ", explanation)

    confidence = interpreter.analyze_decision_confidence(state, action)
    print("\nDecision confidence analysis:")
    for k, v in confidence.items():
        print(f"  {k:25s}: {v}")

    print("\n working")


if __name__ == "__main__":
    test_interpretability()
