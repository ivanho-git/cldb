"""
Deep Q-Network agent for CLDB index selection.

This is the actual reinforcement-learning component: a value-based agent that
maps a state (query features + semantic embedding + workload context +
index-configuration state) to Q-values over the discrete action space
{KEEP, CREATE_INDEX, DROP_INDEX}.

Design notes
------------
* Off-policy DQN with a frozen target network and Huber TD loss.
* Epsilon-greedy exploration with a linear schedule; evaluation is greedy.
* Action masking: structurally invalid actions (e.g. CREATE_INDEX with no
  predicate columns, or DROP_INDEX with no active indexes) are masked out
  before selection, so exploration cannot propose unsafe operations.
* Anti-forgetting: Elastic Weight Consolidation regularises the Q-network,
  with Fisher matrices consolidated per detected workload phase.
* Prioritized replay: sampled with database-aware importance weights, and
  priorities are refined using absolute TD errors.
"""
import copy
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cldb.config import get_config
from cldb.ewc import EWC
from cldb.logger import get_logger
from cldb.replay_buffer import DatabaseAwarePriorityReplay


ACTION_KEEP = 0
ACTION_CREATE = 1
ACTION_DROP = 2
NUM_ACTIONS = 3


class QNetwork(nn.Module):
    """Multi-layer perceptron mapping a state to Q-values for each action."""

    def __init__(self, state_dim: int, num_actions: int = NUM_ACTIONS,
                 hidden_dims: Tuple[int, ...] = (256, 128), dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = state_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_state(features, embedding, context, index_state) -> np.ndarray:
    """Concatenate the four state components into a single flat vector."""
    parts = [np.asarray(features, dtype=np.float32).ravel(),
             np.asarray(embedding, dtype=np.float32).ravel(),
             np.asarray(context, dtype=np.float32).ravel()]
    if index_state is not None:
        parts.append(np.asarray(index_state, dtype=np.float32).ravel())
    return np.concatenate(parts).astype(np.float32)


class DQNAgent:
    def __init__(self, feature_dim: int, embedding_dim: int = 384, context_dim: int = 3,
                 index_state_dim: int = 3, num_actions: int = NUM_ACTIONS,
                 hidden_dims: Tuple[int, ...] = (256, 128), learning_rate: float = 5e-4,
                 gamma: float = 0.97, batch_size: int = 64,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay_steps: int = 3000, target_update_interval: int = 200,
                 ewc_lambda: float = 0.4, device: Optional[str] = None, seed: Optional[int] = None):
        self.logger = get_logger("rl_agent")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        self.index_state_dim = index_state_dim
        self.num_actions = num_actions
        self.state_dim = feature_dim + embedding_dim + context_dim + index_state_dim

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_interval = target_update_interval

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.online = QNetwork(self.state_dim, num_actions, hidden_dims).to(self.device)
        self.target = QNetwork(self.state_dim, num_actions, hidden_dims).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=learning_rate)
        self.ewc = EWC(self.online, lambda_param=ewc_lambda)
        self.steps = 0
        self.updates = 0
        self.cf_updates = 0
        self.td_errors: List[float] = []

    # ------------------------------------------------------------------ acting
    def _forward(self, state: np.ndarray) -> torch.Tensor:
        self.online.eval()
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.online(x).squeeze(0)

    def select_action(self, state: np.ndarray, valid_actions: Optional[np.ndarray] = None,
                      explore: bool = True) -> int:
        """Epsilon-greedy action selection with optional validity masking."""
        q = self._forward(state)
        if valid_actions is not None:
            mask = torch.as_tensor(valid_actions, dtype=torch.bool, device=self.device)
            q = q.masked_fill(~mask, -1e9)

        # Epsilon decays on every interaction, regardless of which branch is taken.
        self.steps += 1
        self._decay_epsilon()

        if explore and np.random.random() < self.epsilon:
            valid_idx = np.flatnonzero(valid_actions) if valid_actions is not None else np.arange(self.num_actions)
            return int(np.random.choice(valid_idx))

        return int(torch.argmax(q).item())

    def act(self, state: np.ndarray, valid_actions: Optional[np.ndarray] = None,
            explore: bool = True) -> Tuple[int, np.ndarray, bool]:
        """Single-forward-pass epsilon-greedy action selection.

        Returns ``(action, raw_q_values, exploring)``. The raw Q-values are
        returned for dashboards; selection itself uses the masked Q-values.
        """
        q = self._forward(state)
        if valid_actions is not None:
            mask = torch.as_tensor(valid_actions, dtype=torch.bool, device=self.device)
            q_masked = q.masked_fill(~mask, -1e9)
        else:
            q_masked = q

        self.steps += 1
        self._decay_epsilon()
        greedy = int(torch.argmax(q_masked).item())

        action = greedy
        if explore and np.random.random() < self.epsilon:
            valid_idx = (np.flatnonzero(valid_actions) if valid_actions is not None
                         else np.arange(self.num_actions))
            action = int(np.random.choice(valid_idx))

        return action, q.detach().cpu().numpy().astype(np.float32), action != greedy

    def predict(self, state: np.ndarray, valid_actions: Optional[np.ndarray] = None):
        """Greedy prediction returning (action, confidence)."""
        q = self._forward(state)
        if valid_actions is not None:
            mask = torch.as_tensor(valid_actions, dtype=torch.bool, device=self.device)
            q = q.masked_fill(~mask, -1e9)
        probs = torch.softmax(q, dim=0)
        confidence, action = torch.max(probs, dim=0)
        return int(action.item()), float(confidence.item())

    def q_values(self, state: np.ndarray,
                 valid_actions: Optional[np.ndarray] = None) -> np.ndarray:
        """Raw Q-values for every action, for inspection and dashboards."""
        q = self._forward(state)
        if valid_actions is not None:
            mask = torch.as_tensor(valid_actions, dtype=torch.bool, device=self.device)
            q = q.masked_fill(~mask, -1e9)
        return q.detach().cpu().numpy().astype(np.float32)

    def _decay_epsilon(self):
        frac = min(1.0, self.steps / max(1, self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def reheat_epsilon(self, target: float) -> float:
        """Rewind the exploration schedule so epsilon restarts at ``target``.

        Called when the workload changes, because the policy learned for the
        previous phase is exactly the one that must now be re-tested.
        """
        target = float(np.clip(target, self.epsilon_end, self.epsilon_start))
        span = self.epsilon_start - self.epsilon_end
        if span <= 0:
            return float(self.epsilon)
        frac = (self.epsilon_start - target) / span
        self.steps = int(round(frac * self.epsilon_decay_steps))
        self.epsilon = target
        return float(self.epsilon)

    # ---------------------------------------------------------------- learning
    def train_from_buffer(self, buffer: DatabaseAwarePriorityReplay,
                          batch_size: Optional[int] = None) -> Optional[Dict]:
        batch_size = batch_size or self.batch_size
        if len(buffer) < max(8, batch_size):
            return None

        batch = buffer.sample_transitions(batch_size)
        states = batch["states"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device)
        weights = batch["weights"].to(self.device)

        self.online.train()
        q_values = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_states).max(1).values
            targets = rewards + self.gamma * (1.0 - dones) * next_q

        td = q_values - targets
        loss_q = (nn.functional.smooth_l1_loss(q_values, targets, reduction="none") * weights).mean()
        ewc_penalty = self.ewc.compute_ewc_loss()
        loss = loss_q + ewc_penalty

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        td_abs = td.detach().abs().cpu().numpy()
        buffer.update_priorities_td(batch["indices"], td_abs.tolist())

        self.updates += 1
        if self.updates % self.target_update_interval == 0:
            self.update_target()

        self.td_errors = td_abs.tolist()
        return {"loss": float(loss.item()), "td_mean": float(td_abs.mean()),
                "ewc_penalty": float(ewc_penalty.item()), "epsilon": float(self.epsilon)}

    def train_counterfactual(self, state: np.ndarray, counterfactuals,
                             weight: float = 1.0, margin: float = 0.1) -> Optional[Dict]:
        """Auxiliary counterfactual preference update from a what-if oracle.

        ``counterfactuals`` is a :class:`cldb.counterfactual.CounterfactualSet`
        holding the one-step outcome of every valid action at ``state``. All
        valid actions are compared through a pairwise ranking loss: the action
        with the best what-if outcome must out-score every strictly worse
        action by ``margin``.

        A ranking objective is used instead of a bootstrapped all-action TD
        target on purpose. Counterfactual targets are *one-step* and dense, so
        bootstrapping them every step makes the value function too plastic and
        interferes with retention; the pairwise form injects the same
        "do not do this" signal while leaving the temporal structure to
        replay/EWC.
        """
        actions = np.asarray(counterfactuals.actions, dtype=np.int64)
        if actions.size == 0:
            return None

        device = self.device
        states = torch.as_tensor(
            np.tile(np.asarray(state, dtype=np.float32), (actions.size, 1)),
            dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=device)
        rewards_t = torch.as_tensor(counterfactuals.rewards, dtype=torch.float32, device=device)

        # Q-values of the valid actions only, shape (k, k): every row is the
        # same state, every column is one of the k candidate actions.
        self.online.train()
        q_all = self.online(states)
        q_valid = q_all[:, actions_t]

        best_idx = int(torch.argmax(rewards_t).item())
        worse = rewards_t < (rewards_t[best_idx] - 1e-6)
        if not bool(worse.any()):
            return None

        q_best = q_valid[:, best_idx:best_idx + 1]
        cf_loss = torch.relu(q_valid[:, worse] - q_best + margin).mean()

        ewc_penalty = self.ewc.compute_ewc_loss()
        loss = weight * cf_loss + ewc_penalty

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        self.updates += 1
        self.cf_updates += 1
        if self.updates % self.target_update_interval == 0:
            self.update_target()

        return {"cf_loss": float(cf_loss.item()),
                "cf_best_reward": float(rewards_t[best_idx].item()),
                "cf_actions": int(actions.size),
                "cf_worse_actions": int(worse.sum().item()),
                "ewc_penalty": float(ewc_penalty.item())}

    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())

    # ------------------------------------------------------------- consolidation
    def compute_fisher(self, states: torch.Tensor, batch_size: int = 128) -> Dict[str, torch.Tensor]:
        """Fisher information from squared gradients of the summed Q-values."""
        fisher = {name: torch.zeros_like(p) for name, p in self.online.named_parameters() if p.requires_grad}
        self.online.eval()
        n_batches = 0
        for i in range(0, max(1, len(states)), batch_size):
            chunk = states[i:i + batch_size].to(self.device)
            if len(chunk) == 0:
                continue
            self.online.zero_grad()
            q = self.online(chunk)
            loss = q.sum(dim=1).mean()
            loss.backward()
            for name, p in self.online.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[name] += p.grad.data ** 2
            n_batches += 1
        for name in fisher:
            fisher[name] /= max(1, n_batches)
        self.online.zero_grad()
        return fisher

    def consolidate(self, phase_id: str, states: torch.Tensor):
        """Register a workload phase for EWC using the current weights."""
        optimal = {name: p.data.clone().detach()
                   for name, p in self.online.named_parameters() if p.requires_grad}
        fisher = self.compute_fisher(states)
        self.ewc.register_phase_matrix(phase_id, fisher, optimal)
        self.logger.info(f"Consolidated phase {phase_id} ({len(self.ewc.optimal_weights)} phases total)")

    # ---------------------------------------------------------------- persistence
    def save(self, path: str):
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "fisher": self.ewc.fisher_matrices,
            "optimal": self.ewc.optimal_weights,
            "state_dim": self.state_dim,
            "num_actions": self.num_actions,
            "steps": self.steps,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.ewc.fisher_matrices = ckpt.get("fisher", {})
        self.ewc.optimal_weights = ckpt.get("optimal", {})
        self.steps = ckpt.get("steps", 0)
        return self
