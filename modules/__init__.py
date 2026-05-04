# DataSarthi AI - Modules Package
# This file makes the modules directory a Python package

try:
    from .data_generator import SyntheticDataGenerator, generate_data
except ImportError:
    pass

try:
    from .system_checker import SystemCompatibilityChecker, check_system
except ImportError:
    pass

try:
    from .model_trainer import ModelTrainer, train_models, apply_pca
except ImportError:
    pass

try:
    from .recommender import ModelRecommender, recommend_model
except ImportError:
    pass

__all__ = [
    'SyntheticDataGenerator',
    'generate_data',
    'SystemCompatibilityChecker', 
    'check_system',
    'ModelTrainer',
    'train_models',
    'apply_pca',
    'ModelRecommender',
    'recommend_model'
]
