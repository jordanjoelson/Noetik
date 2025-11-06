import torch
import numpy as np
from typing import List, Dict, Tuple

class GradientDescentInterpreter:

    def __init__(self, agent):
        self.agent = agent

    def compute_importance(self, state: np.ndarray, action: float) -> Dict[str, float]:
        state_tensor = torch.FloatTensor(state).requires_grad_(True)
        action_tensor = torch.FloatTensor([action])

        q_value = self.agent.q1(state_tensor.unsqueeze(0), action_tensor.unsqueeze(0))
        q_value.backward()
        gradients = state_tensor.grad.abs().detach().numpy()
        
        importance = {
            'user_embedding': float(np.mean(gradients[0:32])),
            'content_preferences': float(np.mean(gradients[32:64])),
            'temporal_patterns': float(np.mean(gradients[64:96])),
            'context_features': float(np.mean(gradients[96:112])),
            'history_embedding': float(np.mean(gradients[112:128]))
        }
        
        return importance

    def get_top_influential_features(self, state: np.ndarray, action: float, top_k: int = 3) -> List[Tuple[str, float]]:
        importance = self.compute_importance(state, action)
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_features[:top_k]
    
    def generate_gradient_explanation(self, state: np.ndarray, action: float) -> str:
        """
        Generate human-readable explanation based on gradient importance
        """
        importance = self.compute_importance(state, action)
        top_features = self.get_top_influential_features(state, action, top_k=2)
        
        # Find the most influential category
        most_influential = max(importance.items(), key=lambda x: x[1])
        
        explanations = {
            'user_embedding': [
                "Based on user's long-term preferences and engagement history",
                "Personalized for this specific user's behavior patterns",
                "Tailored to user's historical click and watch patterns"
            ],
            'content_preferences': [
                "Matches content types this user typically engages with",
                "Similar to videos user has shown preference for",
                "Fits user's content consumption patterns"
            ],
            'temporal_patterns': [
                "Optimized for current engagement trends in this session",
                "Matches the user's recent interaction rhythm",
                "Timed based on current session dynamics"
            ],
            'context_features': [
                "Suitable for current point in the session",
                "Context-aware recommendation for this moment",
                "Adapted to current session progress"
            ],
            'history_embedding': [
                "Consistent with recent interaction history",
                "Builds on previous recommendations in this session",
                "Follows from recent user responses"
            ]
        }
        
        if most_influential[1] > 0.01:
            category_explanations = explanations.get(most_influential[0], [])
            if category_explanations:
                explanation = category_explanations[0]
                
                if len(top_features) > 1 and top_features[1][1] > 0.005:
                    second_category = top_features[1][0]
                    second_explanations = explanations.get(second_category, [])
                    if second_explanations:
                        explanation += f", and {second_explanations[0].lower()}"
                
                return explanation
        
        return "General recommendation based on various factors"
    
    def analyze_decision_confidence(self, state: np.ndarray, action: float, num_alternatives: int = 5) -> Dict:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            chosen_q = self.agent.q1(state_tensor, torch.FloatTensor([[action]]))
            
            alternative_actions = []
            for alt_action in np.linspace(-1, 1, num_alternatives):
                if abs(alt_action - action) > 0.1: #diff from chosen
                    alt_q = self.agent.q1(state_tensor, torch.FloatTensor([[alt_action]]))
                    alternative_actions.append((alt_action, alt_q.item()))
        
        if alternative_actions:
            best_alt_q = max(alternative_actions, key=lambda x: x[1])[1]
            confidence_gap = chosen_q.item() - best_alt_q
            
            return {
                'chosen_q_value': float(chosen_q.item()),
                'best_alternative_q': best_alt_q,
                'confidence_gap': confidence_gap,
                'is_high_confidence': confidence_gap > 0.1,
                'is_ambiguous': confidence_gap < 0.05
            }
        
        return {'chosen_q_value': float(chosen_q.item()), 'confidence_gap': 0.0}

def test_interpretability():
    print("working")