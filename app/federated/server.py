"""
Federated Learning Server
Central coordinator (never sees raw data)
"""

import torch
from typing import List, Dict, Optional
import logging
from pathlib import Path
import json

from app.federated.aggregator import FedAvgAggregator, FedProxAggregator
from app.federated.client import FederatedClient

logger = logging.getLogger(__name__)

class FederatedServer:
    """
    Federated Learning Server
    Coordinates training but never accesses raw data
    """
    
    def __init__(
        self,
        aggregation_method: str = "fedavg",
        save_path: str = "results/models"
    ):
        self.aggregation_method = aggregation_method
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize aggregator
        if aggregation_method == "fedavg":
            self.aggregator = FedAvgAggregator()
        elif aggregation_method == "fedprox":
            self.aggregator = FedProxAggregator(mu=0.01)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation_method}")
        
        self.global_model = None
        self.round_number = 0
        self.training_history = []
        
        logger.info(f"FL Server initialized with {aggregation_method}")
    
    def federated_training(
        self,
        clients: List[FederatedClient],
        rounds: int = 10,
        local_epochs: int = 5,
        client_fraction: float = 1.0
    ) -> Dict:
        """
        Main federated training loop
        
        Args:
            clients: List of federated clients
            rounds: Number of communication rounds
            local_epochs: Local epochs per round
            client_fraction: Fraction of clients to sample per round
        
        Returns:
            Training history and final model
        """
        logger.info(f"Starting federated training: {rounds} rounds, {len(clients)} clients")
        
        num_clients = len(clients)
        
        for round_num in range(1, rounds + 1):
            self.round_number = round_num
            logger.info(f"\n{'='*60}")
            logger.info(f"Round {round_num}/{rounds}")
            logger.info(f"{'='*60}")
            
            # Sample clients
            import random
            num_selected = max(1, int(num_clients * client_fraction))
            selected_clients = random.sample(clients, num_selected)
            
            logger.info(f"Selected {num_selected} clients: {[c.client_id for c in selected_clients]}")
            
            # Client training
            client_models = []
            client_weights = []
            
            for client in selected_clients:
                logger.info(f"\nClient {client.client_id} training...")
                
                # Local training
                local_model = client.local_train(
                    epochs=local_epochs,
                    global_model=self.global_model
                )
                
                client_models.append(local_model)
                client_weights.append(client.data_size)
            
            # Aggregate models
            logger.info(f"\nAggregating {len(client_models)} client models...")
            self.global_model = self.aggregator.aggregate(
                client_models,
                client_weights
            )
            
            # Evaluate
            round_metrics = self._evaluate_round(clients)
            self.training_history.append(round_metrics)
            
            logger.info(f"Round {round_num} complete: Avg accuracy = {round_metrics['avg_accuracy']:.2%}")
            
            # Save checkpoint
            self._save_checkpoint(round_num)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Federated training complete!")
        logger.info(f"{'='*60}")
        
        return {
            'global_model': self.global_model,
            'history': self.training_history
        }
    
    def _evaluate_round(self, clients: List[FederatedClient]) -> Dict:
        """Evaluate all clients after aggregation"""
        accuracies = []
        
        for client in clients:
            # Create test samples (simplified)
            test_samples = [
                {
                    'question': f'Test question for {client.client_id}',
                    'context': 'Test context',
                    'answer': 'Test answer'
                }
                for _ in range(5)
            ]
            
            # Evaluate
            client.apply_global_model(self.global_model)
            metrics = client.evaluate(test_samples)
            accuracies.append(metrics['accuracy'])
        
        return {
            'round': self.round_number,
            'avg_accuracy': sum(accuracies) / len(accuracies),
            'client_accuracies': accuracies
        }
    
    def _save_checkpoint(self, round_num: int):
        """Save model checkpoint"""
        checkpoint_path = self.save_path / f"global_model_round_{round_num}.pt"
        torch.save(self.global_model, checkpoint_path)
        
        # Save history
        history_path = self.save_path / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")