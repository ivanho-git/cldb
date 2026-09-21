"""
Enhanced CL Engine with production-grade features:
- Improved architecture with residual connections
- Batch processing for inference
- Model quantization support
- Caching mechanisms
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Optional, List, Dict
import numpy as np
import hashlib

from cldb.config import get_config
from cldb.logger import get_logger
from cldb.ewc import EWC
from cldb.replay_buffer import DatabaseAwarePriorityReplay


EMBEDDING_DIM = 384
CONTEXT_DIM = 3


class ImprovedPolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.2, use_batch_norm: bool = True):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(x))


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class ResidualPolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.residual1 = ResidualBlock(hidden_dim, dropout)
        self.residual2 = ResidualBlock(hidden_dim, dropout)
        self.output = nn.Linear(hidden_dim, 3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.residual1(x)
        x = self.residual2(x)
        return self.output(x)


class AttentionPolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_proj(x).unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        x = attn_out.squeeze(1)
        return self.output(x)


class EnhancedCLEngine:
    def __init__(self, feature_dim: int, embedding_dim: int = EMBEDDING_DIM, context_dim: int = CONTEXT_DIM,
                 architecture: str = "standard", use_quantization: bool = False, cache_size: int = 1000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        self.total_input_dim = feature_dim + embedding_dim + context_dim
        self.architecture = architecture
        
        config = get_config()
        ml = config.ml
        
        if architecture == "residual":
            self.model = ResidualPolicyNetwork(input_dim=self.total_input_dim, hidden_dim=128, dropout=ml.dropout)
        elif architecture == "attention":
            self.model = AttentionPolicyNetwork(input_dim=self.total_input_dim, hidden_dim=128, num_heads=4, dropout=ml.dropout)
        else:
            self.model = ImprovedPolicyNetwork(input_dim=self.total_input_dim, hidden_dims=ml.hidden_dims, dropout=ml.dropout)
        
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=ml.learning_rate, weight_decay=ml.weight_decay)
        self.criterion = nn.CrossEntropyLoss(reduction='none') # Change to none for IS weights
        self.ewc = EWC(self.model, lambda_param=config.ewc.lambda_param)
        
        self._quantized = False
        if use_quantization and self.device.type == "cpu":
            self._quantize()
        
        self._cache_size = cache_size
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._cache_hits = 0
    
    def _quantize(self):
        try:
            self.model = torch.quantization.quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)
            self._quantized = True
            logger = get_logger("cl_engine")
            logger.info("Model quantized for faster inference")
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
    
    def train_on_batch(self, x_features: torch.Tensor, x_embeddings: torch.Tensor, x_context: torch.Tensor,
                       y_tensor: torch.Tensor, task_id: int = 0):
        self.model.train()
        x_features = x_features.to(self.device)
        x_embeddings = x_embeddings.to(self.device)
        x_context = x_context.to(self.device)
        y_tensor = y_tensor.to(self.device)
        x_combined = torch.cat([x_features, x_embeddings, x_context], dim=1)
        self.optimizer.zero_grad()
        outputs = self.model(x_combined)
        
        loss_batch = self.criterion(outputs, y_tensor)
        loss = loss_batch.mean()
        
        # Add EWC penalty if applicable
        ewc_penalty = self.ewc.compute_ewc_loss()
        total_loss = loss + ewc_penalty
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return total_loss.item()
        
    def train_on_experience(self, replay_buffer: DatabaseAwarePriorityReplay, batch_size: int = 32) -> Tuple[float, float]:
        """
        Trains the model using samples from the Priority Replay Buffer.
        Also applies EWC penalty to prevent catastrophic forgetting.
        Returns:
            Tuple of (total_loss, ewc_penalty)
        """
        if len(replay_buffer) < batch_size:
            return 0.0, 0.0
            
        self.model.train()
        
        # Sample from replay buffer
        feat, emb, ctx, actions, weights, indices = replay_buffer.sample(batch_size)
        
        feat = feat.to(self.device)
        emb = emb.to(self.device)
        ctx = ctx.to(self.device)
        actions = actions.to(self.device)
        weights = weights.to(self.device)
        
        x_combined = torch.cat([feat, emb, ctx], dim=1)
        
        self.optimizer.zero_grad()
        outputs = self.model(x_combined)
        
        # Calculate loss with Importance Sampling weights
        loss_batch = self.criterion(outputs, actions)
        loss = (loss_batch * weights).mean()
        
        # Add EWC penalty
        ewc_penalty = self.ewc.compute_ewc_loss()
        
        total_loss = loss + ewc_penalty
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Note: the reward update should happen in production pipeline after execution,
        # but we could update priority here if we were using TD-errors. Since priority
        # is database-aware (latency/impact), we don't update it from loss.
        
        return total_loss.item(), ewc_penalty.item()
    
    def train_with_dataloader(self, features: torch.Tensor, embeddings: torch.Tensor, context: torch.Tensor,
                              labels: torch.Tensor, batch_size: int = 32, epochs: int = 5) -> Dict:
        dataset = TensorDataset(features, embeddings, context, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        history = {"loss": [], "accuracy": []}
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            for batch_feat, batch_emb, batch_ctx, batch_labels in loader:
                loss = self.train_on_batch(batch_feat, batch_emb, batch_ctx, batch_labels)
                total_loss += loss
                with torch.no_grad():
                    x = torch.cat([batch_feat, batch_emb, batch_ctx], dim=1).to(self.device)
                    outputs = self.model(x)
                    _, predicted = outputs.max(1)
                    total += batch_labels.size(0)
                    correct += predicted.eq(batch_labels.to(self.device)).sum().item()
            avg_loss = total_loss / len(loader)
            accuracy = 100.0 * correct / total
            history["loss"].append(avg_loss)
            history["accuracy"].append(accuracy)
        return history
    
    def _get_cache_key(self, x_features: torch.Tensor, x_embeddings: torch.Tensor, x_context: torch.Tensor) -> str:
        combined = torch.cat([x_features, x_embeddings, x_context], dim=1)
        combined_np = combined.cpu().numpy()
        return hashlib.md5(combined_np.tobytes()).hexdigest()
    
    def predict(self, x_features: torch.Tensor, x_embeddings: torch.Tensor, x_context: torch.Tensor,
                use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        if use_cache and len(self._cache) < self._cache_size:
            cache_key = self._get_cache_key(x_features, x_embeddings, x_context)
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key]
        
        with torch.no_grad():
            x_combined = torch.cat([x_features.to(self.device), x_embeddings.to(self.device), x_context.to(self.device)], dim=1)
            logits = self.model(x_combined)
            probs = torch.softmax(logits, dim=1)
            confidence, action_idx = torch.max(probs, dim=1)
        
        action_idx_np = action_idx.cpu().numpy()
        confidence_np = confidence.cpu().numpy()
        
        if use_cache and len(self._cache) < self._cache_size:
            self._cache[cache_key] = (action_idx_np, confidence_np)
        
        return action_idx_np, confidence_np
    
    def predict_batch(self, features_list: List[torch.Tensor], embeddings_list: List[torch.Tensor],
                      context_list: List[torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        x_features = torch.stack(features_list)
        x_embeddings = torch.stack(embeddings_list)
        x_context = torch.stack(context_list)
        return self.predict(x_features, x_embeddings, x_context)
    
    def predict_from_features_only(self, x_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            zeros_emb = torch.zeros(x_tensor.size(0), self.embedding_dim, device=self.device)
            zeros_ctx = torch.zeros(x_tensor.size(0), self.context_dim, device=self.device)
            x_combined = torch.cat([x_tensor.to(self.device), zeros_emb, zeros_ctx], dim=1)
            logits = self.model(x_combined)
            probs = torch.softmax(logits, dim=1)
            confidence, action_idx = torch.max(probs, dim=1)
        return action_idx.cpu().numpy(), confidence.cpu().numpy()
    
    def save(self, path: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "feature_dim": self.feature_dim,
            "embedding_dim": self.embedding_dim,
            "context_dim": self.context_dim,
            "architecture": self.architecture
        }, path)
    
    @classmethod
    def load(cls, path: str, **kwargs) -> 'EnhancedCLEngine':
        checkpoint = torch.load(path, map_location='cpu')
        engine = cls(
            feature_dim=checkpoint["feature_dim"],
            embedding_dim=checkpoint["embedding_dim"],
            context_dim=checkpoint["context_dim"],
            architecture=checkpoint.get("architecture", "standard"),
            **kwargs
        )
        engine.model.load_state_dict(checkpoint["model_state_dict"])
        engine.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return engine
    
    def get_cache_stats(self) -> Dict:
        return {"cache_size": len(self._cache), "cache_hits": self._cache_hits,
                "hit_rate": self._cache_hits / max(1, len(self._cache) + self._cache_hits)}
    
    def clear_cache(self):
        self._cache.clear()
        self._cache_hits = 0


CLEngine = EnhancedCLEngine