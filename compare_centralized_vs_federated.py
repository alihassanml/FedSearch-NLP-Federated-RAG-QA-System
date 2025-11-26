
import logging
from app.experiments.run_federated import FederatedExperiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_comparison():
    """Compare centralized vs federated approaches"""
    
    print("\n" + "="*70)
    print("Comparing: Centralized vs Federated RAG")
    print("="*70 + "\n")
    
    results = {}
    
    # Experiment 1: Centralized (baseline)
    logger.info("Running centralized baseline...")
    experiment = FederatedExperiment()
    
    # Modify config for centralized
    experiment.config['clients']['use_dp'] = False
    experiment.config['training']['rounds'] = 1  # No federation
    
    centralized_results, centralized_metrics = experiment.run_experiment("centralized_baseline")
    results['centralized'] = {
        'accuracy': centralized_metrics['accuracy']['mean_accuracy'],
        'privacy': 0,  # No privacy
        'communication': 0  # No communication
    }
    
    # Experiment 2: Federated with DP
    logger.info("\nRunning federated with DP...")
    experiment = FederatedExperiment()
    
    experiment.config['clients']['use_dp'] = True
    experiment.config['training']['rounds'] = 10
    
    federated_results, federated_metrics = experiment.run_experiment("federated_with_dp")
    results['federated'] = {
        'accuracy': federated_metrics['accuracy']['mean_accuracy'],
        'privacy': federated_metrics['privacy_cost']['mean_epsilon'],
        'communication': federated_metrics['communication_cost']['total_communication_mb']
    }
    
    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    print("\nCentralized Baseline:")
    print(f"  Accuracy: {results['centralized']['accuracy']:.2%}")
    print(f"  Privacy: None (all data shared)")
    print(f"  Communication: None (centralized)")
    
    print("\nFederated with DP:")
    print(f"  Accuracy: {results['federated']['accuracy']:.2%}")
    print(f"  Privacy: ε = {results['federated']['privacy']:.4f} (protected)")
    print(f"  Communication: {results['federated']['communication']:.2f} MB")
    
    # Calculate tradeoffs
    accuracy_loss = results['centralized']['accuracy'] - results['federated']['accuracy']
    print("\nPrivacy-Utility Tradeoff:")
    print(f"  Accuracy loss: {accuracy_loss:.2%}")
    print(f"  Privacy gained: ε = {results['federated']['privacy']:.4f}")
    print(f"  Communication cost: {results['federated']['communication']:.2f} MB")
    
    print("\n✅ Comparison complete!")
    print("Results saved to:")
    print("  - results/centralized_baseline/")
    print("  - results/federated_with_dp/")


if __name__ == "__main__":
    run_comparison()