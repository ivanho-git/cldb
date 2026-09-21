"""
Multi-objective Reward Function for CLDB.
Accounts for latency improvement, resource usage, and penalties for harmful regressions.
"""
from typing import Dict, Any

class RewardCalculator:
    def __init__(self, latency_weight: float = 1.0, 
                 regression_penalty_weight: float = 2.0,
                 resource_penalty: float = 0.1):
        """
        Args:
            latency_weight: Weight given to positive latency improvements.
            regression_penalty_weight: Extra penalty weight if an action made performance worse.
            resource_penalty: Small constant penalty for actions that consume resources (e.g., CREATE_INDEX).
        """
        self.latency_weight = latency_weight
        self.regression_penalty_weight = regression_penalty_weight
        self.resource_penalty = resource_penalty
        
    def compute_reward(self, 
                       action: str, 
                       latency_before_ms: float, 
                       latency_after_ms: float, 
                       estimated_size_mb: float = 0.0) -> float:
        """
        Computes a normalized reward.
        
        Formula:
            Delta = latency_before - latency_after
            If Delta > 0 (improvement): Reward = Delta * latency_weight
            If Delta < 0 (regression): Reward = Delta * regression_penalty_weight
            
            Cost = resource_penalty if action == 'CREATE_INDEX' else 0
            
            Final Reward = Reward - Cost
            
        Normalizes by dividing by max(latency_before, 1) to get relative improvement percentage.
        """
        if latency_before_ms <= 0:
            return 0.0
            
        # Raw difference in ms
        delta_ms = latency_before_ms - latency_after_ms
        
        # Relative difference (-1.0 to 1.0)
        relative_delta = delta_ms / latency_before_ms
        
        if relative_delta > 0:
            reward = relative_delta * self.latency_weight
        else:
            # Penalize regressions more heavily
            reward = relative_delta * self.regression_penalty_weight
            
        # Resource penalty (e.g., maintaining an index is not free)
        if action == "CREATE_INDEX":
            # For this prototype, we'll apply a flat penalty. In production, 
            # this would scale with estimated_size_mb.
            reward -= self.resource_penalty
            
        # Bound the reward between -2.0 and 2.0 to stabilize learning
        reward = max(min(reward, 2.0), -2.0)
        
        return reward
