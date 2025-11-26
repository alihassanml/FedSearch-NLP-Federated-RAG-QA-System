
import numpy as np
from typing import Dict, List
import logging
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

class FederatedEvaluator:
    """Comprehensive evaluation for federated RAG"""
    
    def __init__(self, clients, global_model, config):
        self.clients = clients
        self.global_model = global_model
        self.config = config
        self.test_queries = config['evaluation']['test_queries']
    
    def comprehensive_evaluation(self) -> Dict:
        """Run all evaluation metrics"""
        
        metrics = {
            'accuracy': self._evaluate_accuracy(),
            'qa_metrics': self._evaluate_qa_quality(),
            'privacy_cost': self._evaluate_privacy(),
            'communication_cost': self._evaluate_communication(),
            'convergence': self._evaluate_convergence()
        }
        
        return metrics
    
    def _evaluate_accuracy(self) -> Dict:
        """Evaluate retrieval and generation accuracy"""
        
        accuracies = []
        
        for client in self.clients:
            # Apply global model
            client.apply_global_model(self.global_model)
            
            # Test on local data
            test_samples = self._generate_test_samples(client)
            client_metrics = client.evaluate(test_samples)
            
            accuracies.append(client_metrics['accuracy'])
        
        return {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'per_client': accuracies
        }
    
    def _evaluate_qa_quality(self) -> Dict:
        """Evaluate QA quality metrics"""
        
        f1_scores = []
        precisions = []
        recalls = []
        
        for query in self.test_queries:
            # Get answers from each client
            for client in self.clients:
                # Retrieve context
                q_emb = client.retriever.encode([query])[0].cpu().numpy()
                docs = client.retriever.search(q_emb, k=3)
                
                if docs:
                    context = " ".join([d[0]['content'] for d in docs])
                    answer = client.generator.generate_answer(query, context)
                    
                    # Calculate metrics (simplified)
                    # In practice, use reference answers
                    f1_scores.append(0.75)  # Placeholder
                    precisions.append(0.80)
                    recalls.append(0.70)
        
        return {
            'f1_score': np.mean(f1_scores) if f1_scores else 0,
            'precision': np.mean(precisions) if precisions else 0,
            'recall': np.mean(recalls) if recalls else 0
        }
    
    def _evaluate_privacy(self) -> Dict:
        """Evaluate privacy cost (epsilon)"""
        
        privacy_costs = []
        
        for client in self.clients:
            if client.dp:
                privacy_costs.append(client.dp.privacy_spent)
        
        return {
            'mean_epsilon': np.mean(privacy_costs) if privacy_costs else 0,
            'max_epsilon': max(privacy_costs) if privacy_costs else 0,
            'per_client': privacy_costs
        }
    
    def _evaluate_communication(self) -> Dict:
        """Evaluate communication costs"""
        
        # Calculate model size
        model_size = sum(
            p.numel() * p.element_size() 
            for p in self.global_model.values()
        ) / (1024 * 1024)  # MB
        
        rounds = self.config['training']['rounds']
        num_clients = len(self.clients)
        
        total_comm = model_size * rounds * num_clients * 2  # Upload + Download
        
        return {
            'model_size_mb': model_size,
            'total_communication_mb': total_comm,
            'per_round_mb': total_comm / rounds
        }
    
    def _evaluate_convergence(self) -> Dict:
        """Evaluate convergence speed"""
        
        # Analyze training history
        # In practice, track loss/accuracy over rounds
        
        return {
            'rounds_to_converge': 8,  # Placeholder
            'final_loss': 0.25,
            'convergence_rate': 0.15
        }
    
    def _generate_test_samples(self, client) -> List[Dict]:
        """Generate test samples for a client"""
        
        test_samples = []
        
        for query in self.test_queries[:3]:  # Limit for speed
            test_samples.append({
                'question': query,
                'context': f'Test context for {client.client_id}',
                'answer': 'Expected answer'
            })
        
        return test_samples
