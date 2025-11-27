"""
Federated Aggregation Algorithms
"""

import torch
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class FedAvgAggregator:
    """Federated Averaging (FedAvg) aggregation"""
    
    def __init__(self, secure: bool = True):
        self.secure = secure
        self.round_number = 0
    
    def aggregate(
        self,
        client_models: List[Dict],
        client_weights: List[float] = None
    ) -> Dict:
        """
        Aggregate client models using weighted average
        
        Args:
            client_models: List of model state_dicts from clients
            client_weights: Optional weights (e.g., by data size)
        
        Returns:
            Aggregated global model state_dict
        """
        if not client_models:
            raise ValueError("No client models to aggregate")
        
        self.round_number += 1
        logger.info(f"Round {self.round_number}: Aggregating {len(client_models)} clients")
        
        # Default: equal weights
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)
        
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # Initialize global model with zeros
        global_model = {}
        
        # Get all parameter names from first client
        param_names = client_models[0].keys()
        
        # Weighted average for each parameter
        for param_name in param_names:
            # Stack all client parameters
            params = torch.stack([
                client_models[i][param_name] * client_weights[i]
                for i in range(len(client_models))
            ])
            
            # Sum across clients
            global_model[param_name] = params.sum(dim=0)
        
        logger.info(f"Aggregation complete for round {self.round_number}")
        
        return global_model
    
    def secure_aggregate(
        self,
        encrypted_models: List[bytes],
        client_weights: List[float] = None
    ) -> Dict:
        """
        Secure aggregation with encrypted model updates
        (Simplified implementation)
        """
        # In practice, use secure multi-party computation
        # This is a placeholder for the concept
        
        logger.info("Performing secure aggregation...")
        
        # Decrypt (in real implementation, use MPC)
        client_models = [self._decrypt(model) for model in encrypted_models]
        
        # Regular aggregation on decrypted models
        return self.aggregate(client_models, client_weights)
    
    def _decrypt(self, encrypted_model: bytes) -> Dict:
        """Placeholder for decryption"""
        # In practice, implement proper decryption
        return encrypted_model


class FedProxAggregator(FedAvgAggregator):
    """FedProx: Handles heterogeneous data better"""
    
    def __init__(self, mu: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu  # Proximal term
    
    def aggregate(self, client_models: List[Dict], **kwargs) -> Dict:
        """FedProx aggregation with proximal term"""
        # Use FedAvg but clients use proximal term during training
        return super().aggregate(client_models, **kwargs)


class ScaffoldAggregator(FedAvgAggregator):
    """SCAFFOLD: Uses control variates for better convergence"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.control_variates = {}
    
    def aggregate(
        self,
        client_models: List[Dict],
        client_controls: List[Dict] = None,
        **kwargs
    ) -> Dict:
        """SCAFFOLD aggregation with control variates"""
        
        # Update control variates
        if client_controls:
            self._update_controls(client_controls)
        
        # Regular aggregation
        return super().aggregate(client_models, **kwargs)
    
    def _update_controls(self, client_controls: List[Dict]):
        """Update global control variates"""
        for key in client_controls[0].keys():
            controls = [c[key] for c in client_controls]
            self.control_variates[key] = torch.stack(controls).mean(dim=0)