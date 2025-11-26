
import logging
from app.federated.client import FederatedClient
from app.federated.server import FederatedServer

logging.basicConfig(level=logging.INFO)

def quick_test():
    """Quick test of FL components"""
    
    print("\n🧪 Quick FL Test\n")
    
    # Test 1: Client initialization
    print("Test 1: Initializing clients...")
    try:
        client1 = FederatedClient(
            client_id="test_hr",
            data_path="data/department_hr",
            retriever_model="sentence-transformers/paraphrase-MiniLM-L3-v2",
            generator_model="google/flan-t5-small",
            use_dp=True,
            epsilon=1.0
        )
        print("✓ Client initialized successfully")
    except Exception as e:
        print(f"✗ Client initialization failed: {e}")
        return
    
    # Test 2: Data loading
    print("\nTest 2: Loading data...")
    try:
        client1.load_local_data()
        print(f"✓ Loaded {client1.data_size} documents")
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return
    
    # Test 3: Local training
    print("\nTest 3: Local training (1 epoch)...")
    try:
        model_update = client1.local_train(epochs=1)
        print(f"✓ Training complete, {len(model_update)} parameters")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return
    
    # Test 4: Server aggregation
    print("\nTest 4: Server aggregation...")
    try:
        server = FederatedServer()
        global_model = server.aggregator.aggregate([model_update])
        print(f"✓ Aggregation complete")
    except Exception as e:
        print(f"✗ Aggregation failed: {e}")
        return
    
    print("\n✅ All tests passed!")
    print("\nReady to run full experiment:")
    print("  python run_complete_experiment.py")


if __name__ == "__main__":
    quick_test()