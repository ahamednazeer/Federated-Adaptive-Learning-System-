"""
Continual Learning Module
Implements Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting
"""

import torch
import torch.nn as nn
import copy
from typing import Dict, List, Optional

class EWC:
    """
    Elastic Weight Consolidation (EWC) for continual learning
    """
    
    def __init__(self, model: nn.Module, fisher_multiplier: float = 1000.0):
        """
        Args:
            model: The neural network model
            fisher_multiplier: Importance weight (lambda) for the EWC penalty
        """
        self.model = model
        self.fisher_multiplier = fisher_multiplier
        self.fisher_matrix = {}
        self.optpar = {}
        
    def register_task(self, dataloader, device='cpu', num_batches=100):
        """
        Compute Fisher Information Matrix for the current task
        to constrain weights important for this task in future training.
        
        Args:
            dataloader: DataLoader for the current task
            device: Computing device
            num_batches: Limit batches to save time
        """
        self.model.eval()
        self.fisher_matrix = {}
        self.optpar = {}
        
        # Initialize Fisher matrix with zeros
        for name, param in self.model.named_parameters():
             if param.requires_grad:
                self.fisher_matrix[name] = torch.zeros_like(param.data)
                self.optpar[name] = param.data.clone()
        
        # Compute Fisher matrix
        count = 0
        for batch in dataloader:
            if count >= num_batches:
                break
            count += 1
            
            self.model.zero_grad()
            
            # Unpack batch (adapting to dictionary format from server.py)
            if isinstance(batch, dict):
                inputs = batch['inputs'].to(device)
                labels = batch['labels'].to(device)
            else:
                inputs, labels = batch[0].to(device), batch[1].to(device)
            
            output = self.model(inputs)
            
            # For classification, we sample from the distribution or use max
            # Standard EWC uses log likelihood of the data
            # Here we approximate with cross entropy loss gradient
            # Assuming output is logits
            
            # Calculate loss
            # Assuming output is logits for simplicity
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, labels)
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_matrix[name] += param.grad.data ** 2 / count
        
        # Normalize
        # (Already averaged implicitly by accumulation strategy if we divided by N, 
        # but here we just accumulated. Let's stick to the simple accumulation 
        # consistent with online EWC implementations or divide by count)
        # Proper Fisher matches expected 2nd derivative.
        
        for name in self.fisher_matrix:
            self.fisher_matrix[name] /= count

    def penalty_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Calculate EWC penalty loss
        
        Args:
            model: Current model being trained
            
        Returns:
            Scalar tensor representing the penalty
        """
        loss = 0
        for name, param in model.named_parameters():
            if name in self.fisher_matrix:
                fisher = self.fisher_matrix[name]
                opt_param = self.optpar[name]
                
                # EWC Loss: sum(F_i * (theta_i - theta*_i)^2)
                loss += (fisher * (param - opt_param).pow(2)).sum()
        
        return loss * (self.fisher_multiplier / 2)

def create_ewc_handler(model: nn.Module, lambda_ewc: float = 1000.0) -> EWC:
    return EWC(model, lambda_ewc)
