
import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.experiments.run_federated import run_federated_experiment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Main experiment execution"""
    
    print("\n" + "="*70)
    print("FedSearch-NLP: Federated Learning Experiment")
    print("="*70 + "\n")
    
    # Check prerequisites
    logger.info("Checking prerequisites...")
    
    # Check if department data exists
    required_dirs = [
        "data/department_hr",
        "data/department_it",
        "data/department_legal"
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            logger.error(f"Missing directory: {dir_path}")
            logger.error("\nPlease run: python setup_department_data.py")
            return
    
    logger.info("✓ Prerequisites check passed")
    
    # Check config
    config_path = "configs/federated_config.yaml"
    if not os.path.exists(config_path):
        logger.warning(f"Config not found: {config_path}")
        logger.info("Using default configuration")
        config_path = None
    else:
        logger.info(f"✓ Using config: {config_path}")
    
    # Run experiment
    logger.info("\nStarting federated learning experiment...")
    
    try:
        results, metrics = run_federated_experiment(config_path)
        
        # Print summary
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE")
        print("="*70)
        print("\nResults Summary:")
        print(f"  Final Accuracy: {metrics['accuracy']['mean_accuracy']:.2%}")
        print(f"  Privacy Cost (ε): {metrics['privacy_cost']['mean_epsilon']:.4f}")
        print(f"  Communication: {metrics['communication_cost']['total_communication_mb']:.2f} MB")
        print("\nResults saved to: results/federated_rag/")
        print("  - training_history.json")
        print("  - evaluation_metrics.json")
        print("  - plots/")
        print("  - models/")
        
    except Exception as e:
        logger.error(f"\nExperiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✅ Experiment completed successfully!")


if __name__ == "__main__":
    main()
