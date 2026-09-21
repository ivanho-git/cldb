"""
Production-grade training module for CLDB with data augmentation,
normalization, checkpointing, and hyperparameter tuning.
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import random

from cldb.config import MLConfig, get_config
from cldb.logger import get_logger


@dataclass
class Checkpoint:
    path: str
    epoch: int
    task_id: int
    loss: float
    accuracy: float
    timestamp: str
    config: Dict
    metrics: Dict


class Normalizer:
    def __init__(self, method: str = "standard"):
        self.method = method
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.min: Optional[np.ndarray] = None
        self.max: Optional[np.ndarray] = None
        self._fitted = False
    
    def fit(self, data: np.ndarray):
        self._fitted = True
        if self.method == "standard":
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0) + 1e-8
        elif self.method == "minmax":
            self.min = np.min(data, axis=0)
            self.max = np.max(data, axis=0)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Normalizer not fitted")
        if self.method == "standard":
            return (data - self.mean) / self.std
        elif self.method == "minmax":
            return (data - self.min) / (self.max - self.min + 1e-8)
        return data
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)
    
    def save(self, path: str):
        params = {"method": self.method, "_fitted": self._fitted}
        if self.mean is not None:
            params["mean"] = self.mean.tolist()
            params["std"] = self.std.tolist()
        if self.min is not None:
            params["min"] = self.min.tolist()
            params["max"] = self.max.tolist()
        with open(path, 'w') as f:
            json.dump(params, f)
    
    @classmethod
    def load(cls, path: str) -> 'Normalizer':
        with open(path, 'r') as f:
            params = json.load(f)
        normalizer = cls(method=params["method"])
        normalizer._fitted = params["_fitted"]
        if "mean" in params:
            normalizer.mean = np.array(params["mean"])
            normalizer.std = np.array(params["std"])
        if "min" in params:
            normalizer.min = np.array(params["min"])
            normalizer.max = np.array(params["max"])
        return normalizer


class TrainingDataset(Dataset):
    def __init__(self, features: torch.Tensor, embeddings: torch.Tensor, context: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.embeddings = embeddings
        self.context = context
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (self.features[idx], self.embeddings[idx], self.context[idx], self.labels[idx])


class ModelCheckpointManager:
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self._checkpoints: List[Checkpoint] = []
    
    def save(self, model: nn.Module, optimizer: optim.Optimizer, epoch: int, task_id: int, metrics: Dict, config: Dict) -> str:
        timestamp = datetime.now(timezone.utc).isoformat()
        checkpoint_name = f"checkpoint_task{task_id}_epoch{epoch}_{timestamp.replace(':', '-')}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "task_id": task_id,
            "metrics": metrics,
            "config": config,
            "timestamp": timestamp
        }
        torch.save(checkpoint_data, checkpoint_path)
        checkpoint = Checkpoint(path=str(checkpoint_path), epoch=epoch, task_id=task_id, loss=metrics.get("val_loss", 0),
                                accuracy=metrics.get("val_accuracy", 0), timestamp=timestamp, config=config, metrics=metrics)
        self._checkpoints.append(checkpoint)
        self._cleanup_old_checkpoints()
        return str(checkpoint_path)
    
    def load(self, path: str, model: nn.Module, optimizer: Optional[optim.Optimizer] = None):
        checkpoint_data = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint_data["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        return checkpoint_data
    
    def get_latest(self, task_id: Optional[int] = None) -> Optional[str]:
        checkpoints = self._checkpoints
        if task_id is not None:
            checkpoints = [c for c in checkpoints if c.task_id == task_id]
        if not checkpoints:
            paths = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"), key=lambda p: p.stat().st_mtime)
            if paths:
                return str(paths[-1])
            return None
        return max(checkpoints, key=lambda c: c.timestamp).path
    
    def _cleanup_old_checkpoints(self):
        while len(self._checkpoints) > self.max_checkpoints:
            oldest = self._checkpoints.pop(0)
            try:
                Path(oldest.path).unlink()
            except Exception:
                pass


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False


@dataclass
class TrainingMetrics:
    epoch: int
    task_id: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    timestamp: str


class Trainer:
    def __init__(self, model: nn.Module, config: Optional[MLConfig] = None, device: Optional[torch.device] = None, logger=None):
        self.config = config or get_config().ml
        self.logger = logger or get_logger("trainer")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.plateau_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=self.config.scheduler_factor,
                                                   patience=self.config.scheduler_patience, min_lr=self.config.min_lr)
        self.criterion = nn.CrossEntropyLoss()
        self.checkpoint_manager = ModelCheckpointManager(self.config.checkpoint_dir or "/app/models/checkpoints")
        self.early_stopping = EarlyStopping(patience=self.config.early_stopping_patience, mode="min")
        self.history: List[TrainingMetrics] = []
    
    def train_epoch(self, train_loader: DataLoader, task_id: int = 0) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_features, batch_embeddings, batch_context, batch_labels in train_loader:
            batch_features = batch_features.to(self.device)
            batch_embeddings = batch_embeddings.to(self.device)
            batch_context = batch_context.to(self.device)
            batch_labels = batch_labels.to(self.device)
            x_combined = torch.cat([batch_features, batch_embeddings, batch_context], dim=1)
            self.optimizer.zero_grad()
            outputs = self.model(x_combined)
            loss = self.criterion(outputs, batch_labels)
            loss.backward()
            if self.config.gradient_clip_value > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
            self.optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_features, batch_embeddings, batch_context, batch_labels in val_loader:
                batch_features = batch_features.to(self.device)
                batch_embeddings = batch_embeddings.to(self.device)
                batch_context = batch_context.to(self.device)
                batch_labels = batch_labels.to(self.device)
                x_combined = torch.cat([batch_features, batch_embeddings, batch_context], dim=1)
                outputs = self.model(x_combined)
                loss = self.criterion(outputs, batch_labels)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def fit(self, features: torch.Tensor, embeddings: torch.Tensor, context: torch.Tensor, labels: torch.Tensor,
            task_id: int = 0, val_split: float = 0.2) -> Dict:
        n_samples = len(labels)
        indices = torch.randperm(n_samples)
        val_size = int(n_samples * val_split)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        train_dataset = TensorDataset(features[train_indices], embeddings[train_indices], context[train_indices], labels[train_indices])
        val_dataset = TensorDataset(features[val_indices], embeddings[val_indices], context[val_indices], labels[val_indices])
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
        
        self.logger.info(f"Starting training for task {task_id}", train_samples=len(train_indices), val_samples=len(val_indices))
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch(train_loader, task_id)
            val_loss, val_acc = self.validate(val_loader)
            self.plateau_scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            metrics_record = TrainingMetrics(epoch=epoch, task_id=task_id, train_loss=train_loss, train_accuracy=train_acc,
                                              val_loss=val_loss, val_accuracy=val_acc, learning_rate=current_lr,
                                              timestamp=datetime.now(timezone.utc).isoformat())
            self.history.append(metrics_record)
            
            self.logger.info(f"Epoch {epoch+1}/{self.config.epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                             f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.checkpoint_manager.save(self.model, self.optimizer, epoch, task_id, {"val_loss": val_loss, "val_accuracy": val_acc}, {"lr": current_lr})
            
            if self.early_stopping(val_loss):
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return self.get_history()
    
    def get_history(self) -> Dict:
        return {"epochs": len(self.history), "final_train_loss": self.history[-1].train_loss if self.history else None,
                "final_val_loss": self.history[-1].val_loss if self.history else None,
                "final_val_accuracy": self.history[-1].val_accuracy if self.history else None,
                "history": [{"epoch": m.epoch, "task_id": m.task_id, "train_loss": m.train_loss, "train_accuracy": m.train_accuracy,
                             "val_loss": m.val_loss, "val_accuracy": m.val_accuracy, "learning_rate": m.learning_rate} for m in self.history]}
    
    def load_best(self, task_id: Optional[int] = None):
        path = self.checkpoint_manager.get_latest(task_id)
        if path:
            self.checkpoint_manager.load(path, self.model, self.optimizer)
            self.logger.info(f"Loaded checkpoint from {path}")


class HyperparameterTuner:
    def __init__(self, model_class: type, param_space: Dict, n_trials: int = 10, metric: str = "val_accuracy"):
        self.model_class = model_class
        self.param_space = param_space
        self.n_trials = n_trials
        self.metric = metric
        self.results: List[Dict] = []
    
    def suggest_params(self) -> Dict:
        params = {}
        for name, space in self.param_space.items():
            if isinstance(space, list):
                params[name] = random.choice(space)
            elif isinstance(space, tuple) and len(space) == 2:
                if isinstance(space[0], int):
                    params[name] = random.randint(space[0], space[1])
                else:
                    params[name] = random.uniform(space[0], space[1])
        return params
    
    def tune(self, train_data, val_data, fixed_params: Dict = None) -> Dict:
        best_score = 0
        best_params = None
        for trial in range(self.n_trials):
            params = self.suggest_params()
            if fixed_params:
                params.update(fixed_params)
            model = self.model_class(**params)
            trainer = Trainer(model, params)
            history = trainer.fit(train_data, val_data)
            final_score = history.get(f"final_{self.metric}", 0)
            self.results.append({"params": params, "score": final_score})
            if final_score > best_score:
                best_score = final_score
                best_params = params
        return {"best_params": best_params, "best_score": best_score, "all_results": self.results}