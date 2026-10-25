"""
Database-Aware Priority Replay Buffer for CLDB.

Stores (state, action, reward, next_state, done) transitions for off-policy RL,
while retaining the database-aware priority signal used to surface rare but
high-impact workload experiences. Priorities are refined with TD errors after
each learning step (prioritized replay).
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
    index_state: Optional[np.ndarray] = None
    next_features: Optional[np.ndarray] = None
    next_embedding: Optional[np.ndarray] = None
    next_context: Optional[np.ndarray] = None
    next_index_state: Optional[np.ndarray] = None
    done: bool = False
    timestamp: float = field(default_factory=time.time)
    priority: float = 1.0


class DatabaseAwarePriorityReplay:
    def __init__(self, capacity: int = 1000, alpha: float = 0.6, beta: float = 0.4, min_priority: float = 1e-3):
        self.capacity = capacity
        self.alpha = alpha      # Prioritization exponent
        self.beta = beta        # Importance sampling exponent
        self.min_priority = min_priority

        self.buffer: List[Experience] = []
        self._lock = threading.Lock()

        # Track workload frequencies to calculate rarity
        self.action_counts: Dict[int, int] = {}

    def _compute_priority(self, experience: Experience) -> float:
        """
        Initial priority from database-aware characteristics:
        1. Optimization Impact: absolute reward magnitude (latency improvement).
        2. Workload Rarity: inverse frequency of the chosen action.
        TD errors refine this after learning.
        """
        impact = abs(experience.reward) + self.min_priority
        action_freq = self.action_counts.get(experience.action, 1)
        rarity_factor = 1.0 / (action_freq ** 0.5)
        return impact * rarity_factor

    def add(self, query_hash: str, features: np.ndarray, embedding: np.ndarray,
            context: np.ndarray, action: int, reward: float,
            index_state: Optional[np.ndarray] = None,
            next_features: Optional[np.ndarray] = None,
            next_embedding: Optional[np.ndarray] = None,
            next_context: Optional[np.ndarray] = None,
            next_index_state: Optional[np.ndarray] = None,
            done: bool = False):
        with self._lock:
            exp = Experience(query_hash, features, embedding, context, action, reward,
                             index_state=index_state,
                             next_features=next_features, next_embedding=next_embedding,
                             next_context=next_context, next_index_state=next_index_state,
                             done=done)

            self.action_counts[action] = self.action_counts.get(action, 0) + 1
            exp.priority = self._compute_priority(exp)

            if len(self.buffer) < self.capacity:
                self.buffer.append(exp)
            else:
                min_idx = min(range(len(self.buffer)), key=lambda i: self.buffer[i].priority)
                evicted = self.buffer[min_idx]
                self.action_counts[evicted.action] -= 1
                if self.action_counts[evicted.action] <= 0:
                    del self.action_counts[evicted.action]
                self.buffer[min_idx] = exp

    def add_transition(self, query_hash: str, action: int, reward: float, done: bool,
                       state: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                       next_state: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        """Convenience wrapper for RL transitions expressed as (feat, emb, ctx, index_state)."""
        self.add(
            query_hash=query_hash,
            features=state[0], embedding=state[1], context=state[2], index_state=state[3],
            action=action, reward=reward,
            next_features=next_state[0], next_embedding=next_state[1],
            next_context=next_state[2], next_index_state=next_state[3],
            done=done,
        )

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """Legacy sampling of (features, embedding, context, actions, weights, indices)."""
        with self._lock:
            if not self.buffer:
                raise ValueError("Cannot sample from an empty buffer")
            batch_size = min(batch_size, len(self.buffer))

            priorities = np.array([exp.priority for exp in self.buffer], dtype=np.float64)
            probs = priorities ** self.alpha
            probs /= probs.sum()
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False, p=probs)

            total = len(self.buffer)
            weights = (total * probs[indices]) ** (-self.beta)
            weights /= weights.max()

            batch = [self.buffer[i] for i in indices]
            features_t = torch.tensor(np.stack([b.features for b in batch]), dtype=torch.float32)
            embedding_t = torch.tensor(np.stack([b.embedding for b in batch]), dtype=torch.float32)
            context_t = torch.tensor(np.stack([b.context for b in batch]), dtype=torch.float32)
            action_t = torch.tensor([b.action for b in batch], dtype=torch.long)
            weights_t = torch.tensor(weights, dtype=torch.float32)
            return features_t, embedding_t, context_t, action_t, weights_t, indices.tolist()

    def sample_transitions(self, batch_size: int):
        """Sample a full transition batch for off-policy RL.

        Returns dict with state/next_state tensors, actions, rewards, dones,
        importance-sampling weights and the sampled indices.
        """
        with self._lock:
            if not self.buffer:
                raise ValueError("Cannot sample from an empty buffer")
            batch_size = min(batch_size, len(self.buffer))

            priorities = np.array([exp.priority for exp in self.buffer], dtype=np.float64)
            probs = priorities ** self.alpha
            probs /= probs.sum()
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False, p=probs)

            total = len(self.buffer)
            weights = (total * probs[indices]) ** (-self.beta)
            weights /= weights.max()

            batch = [self.buffer[i] for i in indices]

        def _pad(arr, ref):
            if arr is None:
                return np.zeros_like(ref, dtype=np.float32)
            return arr.astype(np.float32)

        def concat(b):
            return np.concatenate([
                b.features.astype(np.float32),
                b.embedding.astype(np.float32),
                b.context.astype(np.float32),
                _pad(b.index_state, b.context),
            ])

        def concat_next(b):
            return np.concatenate([
                _pad(b.next_features, b.features),
                _pad(b.next_embedding, b.embedding),
                _pad(b.next_context, b.context),
                _pad(b.next_index_state, b.context),
            ])

        states = torch.tensor(np.stack([concat(b) for b in batch]), dtype=torch.float32)
        next_states = torch.tensor(np.stack([concat_next(b) for b in batch]), dtype=torch.float32)
        actions = torch.tensor([b.action for b in batch], dtype=torch.long)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32)
        dones = torch.tensor([float(b.done) for b in batch], dtype=torch.float32)
        weights_t = torch.tensor(weights, dtype=torch.float32)
        return {
            "states": states, "next_states": next_states, "actions": actions,
            "rewards": rewards, "dones": dones, "weights": weights_t,
            "indices": indices.tolist(),
        }

    def update_priorities(self, indices: List[int], new_rewards: List[float]):
        with self._lock:
            for idx, reward in zip(indices, new_rewards):
                if idx < len(self.buffer):
                    self.buffer[idx].reward = reward
                    self.buffer[idx].priority = self._compute_priority(self.buffer[idx])

    def update_priorities_td(self, indices: List[int], td_errors: List[float]):
        """Refine priorities using absolute TD errors (Schaul et al. prioritized replay)."""
        with self._lock:
            for idx, td in zip(indices, td_errors):
                if idx < len(self.buffer):
                    exp = self.buffer[idx]
                    rarity = 1.0 / (self.action_counts.get(exp.action, 1) ** 0.5)
                    exp.priority = (abs(float(td)) + self.min_priority) * rarity

    def save(self, filepath: str):
        with self._lock:
            with open(filepath, 'wb') as f:
                pickle.dump(self.buffer, f)

    def load(self, filepath: str):
        with self._lock:
            with open(filepath, 'rb') as f:
                self.buffer = pickle.load(f)
            self.action_counts.clear()
            for exp in self.buffer:
                self.action_counts[exp.action] = self.action_counts.get(exp.action, 0) + 1

    def __len__(self):
        with self._lock:
            return len(self.buffer)
