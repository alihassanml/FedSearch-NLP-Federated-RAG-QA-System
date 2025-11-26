
import os
import json
import logging
from pathlib import Path
from typing import Dict, List
import yaml

from app.federated.client import FederatedClient
from app.federated.server import FederatedServer
from app.experiments.evaluate import FederatedEvaluator
from app.experiments.visualize import plot_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedExperiment:
    """Main experiment orchestrator"""
    
    def __init__(self, config_path: str = "configs/federated_config.yaml"):
        self.config = self._load_config(config_path)
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load experiment configuration"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'clients': {
                    'num_clients': 3,
                    'departments': ['hr', 'it', 'legal'],
                    'use_dp': True,
                    'epsilon': 1.0
                },
                'training': {
                    'rounds': 10,
                    'local_epochs': 5,
                    'batch_size': 4,
                    'learning_rate': 1e-4,
                    'client_fraction': 1.0
                },
                'models': {
                    'retriever': 'sentence-transformers/paraphrase-MiniLM-L3-v2',
                    'generator': 'google/flan-t5-small'
                },
                'evaluation': {
                    'metrics': ['accuracy', 'f1', 'privacy_cost'],
                    'test_queries': [
                        'What is the annual leave policy?',
                        'What is the password policy?',
                        'What compliance certifications exist?'
                    ]
                }
            }
    
    def setup_clients(self) -> List[FederatedClient]:
        """Initialize federated clients (simulating departments)"""
        clients = []
        
        departments = self.config['clients']['departments']
        
        for dept in departments:
            logger.info(f"Setting up client: {dept}")
            
            client = FederatedClient(
                client_id=dept,
                data_path=f"data/department_{dept}",
                retriever_model=self.config['models']['retriever'],
                generator_model=self.config['models']['generator'],
                use_dp=self.config['clients']['use_dp'],
                epsilon=self.config['clients']['epsilon'],
                learning_rate=self.config['training']['learning_rate']
            )
            
            # Load local data
            client.load_local_data()
            
            clients.append(client)
        
        logger.info(f"Initialized {len(clients)} federated clients")
        return clients
    
    def run_experiment(self, experiment_name: str = "federated_rag"):
        """Run complete federated learning experiment"""
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Starting Experiment: {experiment_name}")
        logger.info(f"{'='*70}\n")
        
        # Setup
        clients = self.setup_clients()
        server = FederatedServer(
            aggregation_method="fedavg",
            save_path=f"results/{experiment_name}/models"
        )
        
        # Train
        logger.info("Starting federated training...")
        results = server.federated_training(
            clients=clients,
            rounds=self.config['training']['rounds'],
            local_epochs=self.config['training']['local_epochs'],
            client_fraction=self.config['training']['client_fraction']
        )
        
        # Evaluate
        logger.info("\nEvaluating federated model...")
        evaluator = FederatedEvaluator(
            clients=clients,
            global_model=results['global_model'],
            config=self.config
        )
        
        metrics = evaluator.comprehensive_evaluation()
        
        # Save results
        self._save_results(experiment_name, results, metrics)
        
        # Visualize
        logger.info("\nGenerating visualizations...")
        plot_results(
            history=results['history'],
            metrics=metrics,
            save_dir=f"results/{experiment_name}/plots"
        )
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Experiment Complete: {experiment_name}")
        logger.info(f"Results saved to: results/{experiment_name}/")
        logger.info(f"{'='*70}\n")
        
        return results, metrics
    
    def _save_results(self, experiment_name: str, results: Dict, metrics: Dict):
        """Save experiment results"""
        save_dir = self.results_dir / experiment_name
        save_dir.mkdir(exist_ok=True)
        
        # Save training history
        with open(save_dir / "training_history.json", 'w') as f:
            json.dump(results['history'], f, indent=2)
        
        # Save evaluation metrics
        with open(save_dir / "evaluation_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save configuration
        with open(save_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)
        
        logger.info(f"Results saved to: {save_dir}")


def run_federated_experiment(config_path: str = None):
    """
    Main entry point to run federated experiment
    
    Usage:
        python -m app.experiments.run_federated
    """
    experiment = FederatedExperiment(config_path or "configs/federated_config.yaml")
    results, metrics = experiment.run_experiment()
    return results, metrics


if __name__ == "__main__":
    run_federated_experiment()
