"""
Differential Privacy Implementation using DP-SGD
"""

import torch
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class DifferentialPrivacy:
    """Implements DP-SGD for privacy-preserving training"""
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.privacy_spent = 0.0
        
        logger.info(f"DP initialized: ε={epsilon}, δ={delta}")
    
    def clip_gradients(
        self,
        parameters: List[torch.Tensor],
        max_norm: float = None
    ) -> List[torch.Tensor]:
        """Clip gradients to bound sensitivity"""
        if max_norm is None:
            max_norm = self.max_grad_norm
        
        # Calculate total norm
        total_norm = torch.sqrt(
            sum(p.grad.norm(2).item() ** 2 for p in parameters if p.grad is not None)
        )
        
        # Clip if necessary
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                if p.grad is not None:
                    p.grad.mul_(clip_coef)
        
        return parameters
    
    def add_noise(
        self,
        parameters: List[torch.Tensor],
        sensitivity: float = None
    ) -> List[torch.Tensor]:
        """Add Gaussian noise to gradients"""
        if sensitivity is None:
            sensitivity = self.max_grad_norm
        
        noise_scale = sensitivity * self.noise_multiplier
        
        for p in parameters:
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * noise_scale
                p.grad.add_(noise)
        
        return parameters
    
    def get_privacy_spent(self, steps: int, batch_size: int, data_size: int) -> Dict:
        """Calculate privacy budget spent"""
        # Simplified privacy accounting
        sampling_rate = batch_size / data_size
        self.privacy_spent += self.epsilon * sampling_rate * steps
        
        return {
            'epsilon': self.privacy_spent,
            'delta': self.delta,
            'steps': steps
        }
    
    def track_privacy(self, steps: int):
        """Track cumulative privacy loss"""
        # Using moments accountant (simplified)
        alpha = self.noise_multiplier
        self.privacy_spent += np.sqrt(2 * np.log(1.25 / self.delta)) / alpha
        
        logger.info(f"Privacy spent: ε={self.privacy_spent:.4f}")
        
        return self.privacy_spent