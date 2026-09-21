"""
Database-Aware Priority Replay Buffer for CLDB.
Prioritizes experiences based on workload diversity, optimization impact, 
workload rarity, and historical importance rather than generic ML replay metrics.
"""
import threading
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import time
import pickle


@dataclass
class Experience:
    query_hash: str
    features: np.ndarray
    embedding: np.ndarray
    context: np.ndarray
    action: int
    reward: float
    timestamp: float = field(default_factory=time.time)
    priority: float = 1.0


class DatabaseAwarePriorityReplay:
    def __init__(self, capacity: int = 1000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta    # Importance sampling exponent
        
        self.buffer: List[Experience] = []
        self._lock = threading.Lock()
        
        # Track workload frequencies to calculate rarity
        self.action_counts: Dict[int, int] = {}
        
    def _compute_priority(self, experience: Experience) -> float:
        """
        Computes the priority score of an experience based on database-aware characteristics.
        1. Optimization Impact: Absolute reward magnitude (latency improvement/regression).
        2. Workload Rarity: Inverse frequency of the chosen action/cluster.
        """
        impact = abs(experience.reward) + 1e-5  # Avoid zero priority
        
        # Frequency of this action type
        action_freq = self.action_counts.get(experience.action, 1)
        rarity_factor = 1.0 / (action_freq ** 0.5)
        
        return impact * rarity_factor

    def add(self, query_hash: str, features: np.ndarray, embedding: np.ndarray, 
            context: np.ndarray, action: int, reward: float):
        with self._lock:
            exp = Experience(query_hash, features, embedding, context, action, reward)
            
            # Update frequency tracking
            self.action_counts[action] = self.action_counts.get(action, 0) + 1
            
            exp.priority = self._compute_priority(exp)
            
            if len(self.buffer) < self.capacity:
                self.buffer.append(exp)
            else:
                # Evict the lowest priority experience
                min_idx = min(range(len(self.buffer)), key=lambda i: self.buffer[i].priority)
                
                evicted = self.buffer[min_idx]
                self.action_counts[evicted.action] -= 1
                if self.action_counts[evicted.action] <= 0:
                    del self.action_counts[evicted.action]
                    
                self.buffer[min_idx] = exp
                
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        with self._lock:
            if not self.buffer:
                raise ValueError("Cannot sample from an empty buffer")
                
            batch_size = min(batch_size, len(self.buffer))
            
            # Compute probabilities proportional to priority^alpha
            priorities = np.array([exp.priority for exp in self.buffer], dtype=np.float32)
            probs = priorities ** self.alpha
            probs /= probs.sum()
            
            # Sample indices based on computed probabilities
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False, p=probs)
            
            # Compute Importance Sampling (IS) weights to correct for bias
            # IS_weight = (1 / N * 1 / P(i)) ^ beta
            total = len(self.buffer)
            weights = (total * probs[indices]) ** (-self.beta)
            weights /= weights.max()  # Normalize for stability
            
            batch = [self.buffer[i] for i in indices]
            
            features_t = torch.tensor(np.stack([b.features for b in batch]), dtype=torch.float32)
            embedding_t = torch.tensor(np.stack([b.embedding for b in batch]), dtype=torch.float32)
            context_t = torch.tensor(np.stack([b.context for b in batch]), dtype=torch.float32)
            action_t = torch.tensor([b.action for b in batch], dtype=torch.long)
            weights_t = torch.tensor(weights, dtype=torch.float32)
            
            return features_t, embedding_t, context_t, action_t, weights_t, indices.tolist()
            
    def update_priorities(self, indices: List[int], new_rewards: List[float]):
        """Update priorities if new information changes the reward/impact estimate."""
        with self._lock:
            for idx, reward in zip(indices, new_rewards):
                if idx < len(self.buffer):
                    self.buffer[idx].reward = reward
                    self.buffer[idx].priority = self._compute_priority(self.buffer[idx])
                    
    def save(self, filepath: str):
        with self._lock:
            with open(filepath, 'wb') as f:
                pickle.dump(self.buffer, f)
                
    def load(self, filepath: str):
        with self._lock:
            with open(filepath, 'rb') as f:
                self.buffer = pickle.load(f)
                
            # Reconstruct counts
            self.action_counts.clear()
            for exp in self.buffer:
                self.action_counts[exp.action] = self.action_counts.get(exp.action, 0) + 1
                
    def __len__(self):
        with self._lock:
            return len(self.buffer)
