"""
Native PyTorch Elastic Weight Consolidation (EWC) implementation for CLDB.
Maintains parameter importance across workload phases detected by the drift detector.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy
from typing import Dict, Optional, Tuple


class EWC:
    def __init__(self, model: nn.Module, lambda_param: float = 0.4):
        """
        Args:
            model: The PyTorch neural network to protect.
            lambda_param: How strong the EWC penalty should be.
        """
        self.model = model
        self.lambda_param = lambda_param
        
        self.device = next(model.parameters()).device
        
        # Dictionary to store the optimal weights for each detected workload phase
        self.optimal_weights: Dict[str, torch.Tensor] = {}
        # Dictionary to store the Fisher Information Matrix (parameter importance) for each phase
        self.fisher_matrices: Dict[str, torch.Tensor] = {}
        
    def compute_fisher_matrix(self, dataloader: DataLoader, criterion: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Computes the Fisher Information Matrix (FIM) which estimates the importance
        of each parameter for the current workload phase.
        """
        fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)
                
        self.model.eval()
        
        total_samples = 0
        for batch_features, batch_embeddings, batch_context, batch_labels in dataloader:
            batch_features = batch_features.to(self.device)
            batch_embeddings = batch_embeddings.to(self.device)
            batch_context = batch_context.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            x_combined = torch.cat([batch_features, batch_embeddings, batch_context], dim=1)
            
            self.model.zero_grad()
            outputs = self.model(x_combined)
            
            # Use negative log likelihood for Fisher computation (standard EWC)
            log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
            
            # Expected Fisher: sum over all possible classes weighted by their probability
            probs = torch.exp(log_probs).detach()
            
            for class_idx in range(outputs.size(1)):
                self.model.zero_grad()
                
                # NLL for a specific class
                class_nll = -log_probs[:, class_idx].mean()
                class_nll.backward(retain_graph=True)
                
                # Accumulate the squared gradients weighted by the probability of this class
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # fisher = E[(grad(log(p)))^2]
                        fisher[name] += probs[:, class_idx].mean().item() * (param.grad.data ** 2)
                        
            total_samples += batch_labels.size(0)
            
        # Average over all samples
        for name in fisher:
            fisher[name] /= total_samples
            
        return fisher
        
    def register_workload_phase(self, phase_id: str, features: torch.Tensor, embeddings: torch.Tensor, 
                                context: torch.Tensor, labels: torch.Tensor):
        """
        Registers a new workload phase (detected by the drift detector) by snapshotting 
        the optimal weights and computing the Fisher matrix for this phase.
        """
        dataset = TensorDataset(features, embeddings, context, labels)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Save optimal weights for this phase
        self.optimal_weights[phase_id] = {
            name: param.data.clone().detach() 
            for name, param in self.model.named_parameters() if param.requires_grad
        }
        
        # Compute and store FIM
        criterion = nn.CrossEntropyLoss()
        self.fisher_matrices[phase_id] = self.compute_fisher_matrix(dataloader, criterion)
        
    def compute_ewc_loss(self) -> torch.Tensor:
        """
        Computes the EWC penalty loss to be added to the standard task loss.
        L_total = L_current + lambda/2 * sum_i(F_i * (theta - theta_star_i)^2)
        """
        loss = torch.tensor(0.0, device=self.device)
        
        if not self.optimal_weights:
            return loss
            
        for phase_id in self.optimal_weights:
            fisher = self.fisher_matrices[phase_id]
            optimal = self.optimal_weights[phase_id]
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in fisher:
                    # penalty = F * (theta - theta_star)^2
                    penalty = fisher[name] * (param - optimal[name]) ** 2
                    loss += penalty.sum()
                    
        return (self.lambda_param / 2.0) * loss
