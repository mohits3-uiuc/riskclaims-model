"""
Evaluation Package for Claims Risk Classification Pipeline

This package provides comprehensive model evaluation and comparison capabilities
for the ML pipeline including metrics calculation, statistical testing, and reporting.
"""

from .model_evaluator import ModelEvaluator
from .model_comparison import ModelComparison

__all__ = [
    'ModelEvaluator',
    'ModelComparison'
]

__version__ = "1.0.0"