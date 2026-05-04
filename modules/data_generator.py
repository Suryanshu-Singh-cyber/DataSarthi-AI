# modules/data_generator.py
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

class SyntheticDataGenerator:
    
    @staticmethod
    def generate_classification(n_samples=1000, n_features=10, n_classes=2, 
                                n_informative=8, random_state=42):
        """Generate synthetic classification dataset"""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            random_state=random_state
        )
        
        feature_cols = [f'feature_{i+1}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_cols)
        df['target'] = y
        
        return df
    
    @staticmethod
    def generate_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42):
        """Generate synthetic regression dataset"""
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=random_state
        )
        
        feature_cols = [f'feature_{i+1}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_cols)
        df['target'] = y
        
        return df

# Simple function for easy import
def generate_data(dataset_type="classification", n_samples=1000, n_features=10, noise=0.1):
    """Simple function interface for backward compatibility"""
    if dataset_type == "classification":
        return SyntheticDataGenerator.generate_classification(
            n_samples=n_samples, 
            n_features=n_features
        )
    else:
        return SyntheticDataGenerator.generate_regression(
            n_samples=n_samples, 
            n_features=n_features, 
            noise=noise
        )
