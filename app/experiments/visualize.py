
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List
import json

sns.set_style("whitegrid")

def plot_results(history: List[Dict], metrics: Dict, save_dir: str = "results/plots"):
    """Generate all visualization plots"""
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Training convergence
    plot_training_convergence(history, save_path / "convergence.png")
    
    # Plot 2: Accuracy per client
    plot_client_accuracy(metrics, save_path / "client_accuracy.png")
    
    # Plot 3: Privacy cost
    plot_privacy_cost(metrics, save_path / "privacy_cost.png")
    
    # Plot 4: Communication overhead
    plot_communication(metrics, save_path / "communication.png")
    
    print(f"Plots saved to: {save_path}")


def plot_training_convergence(history: List[Dict], save_path: Path):
    """Plot training accuracy over rounds"""
    
    rounds = [h['round'] for h in history]
    accuracies = [h['avg_accuracy'] for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, accuracies, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Average Accuracy', fontsize=12)
    plt.title('Federated Learning Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_client_accuracy(metrics: Dict, save_path: Path):
    """Plot per-client accuracy"""
    
    accuracies = metrics['accuracy']['per_client']
    clients = [f'Client {i+1}' for i in range(len(accuracies))]
    
    plt.figure(figsize=(10, 6))
    plt.bar(clients, accuracies, color='steelblue', alpha=0.8)
    plt.xlabel('Client', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Per-Client Model Accuracy', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_privacy_cost(metrics: Dict, save_path: Path):
    """Plot privacy budget spent per client"""
    
    if 'privacy_cost' in metrics and metrics['privacy_cost']['per_client']:
        epsilons = metrics['privacy_cost']['per_client']
        clients = [f'Client {i+1}' for i in range(len(epsilons))]
        
        plt.figure(figsize=(10, 6))
        plt.bar(clients, epsilons, color='coral', alpha=0.8)
        plt.axhline(y=1.0, color='red', linestyle='--', label='Target ε=1.0')
        plt.xlabel('Client', fontsize=12)
        plt.ylabel('Privacy Budget (ε)', fontsize=12)
        plt.title('Privacy Cost per Client', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_communication(metrics: Dict, save_path: Path):
    """Plot communication costs"""
    
    comm = metrics['communication_cost']
    
    categories = ['Model Size', 'Per Round', 'Total']
    values = [
        comm['model_size_mb'],
        comm['per_round_mb'],
        comm['total_communication_mb'] / 100  # Scale for visibility
    ]
    
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color=['steelblue', 'coral', 'lightgreen'], alpha=0.8)
    plt.ylabel('Size (MB)', fontsize=12)
    plt.title('Communication Overhead', fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()