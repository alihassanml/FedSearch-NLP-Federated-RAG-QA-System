# ============================================================
# File: app/experiments/__init__.py
# ============================================================

"""Experiment framework for federated learning evaluation"""

from .run_federated import run_federated_experiment
from .evaluate import FederatedEvaluator
from .visualize import plot_results

__all__ = ['run_federated_experiment', 'FederatedEvaluator', 'plot_results']
