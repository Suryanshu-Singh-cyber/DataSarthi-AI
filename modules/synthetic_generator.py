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
        
        # Create DataFrame
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
    
    @staticmethod
    def generate_custom(n_samples=500, feature_types=None, distributions=None):
        """
        Generate custom dataset with specified feature types and distributions
        
        Parameters:
        - n_samples: number of rows
        - feature_types: dict {'feature_name': 'numerical'/'categorical'}
        - distributions: dict {'feature_name': {'type': 'normal', 'mean': 0, 'std': 1}}
        """
        data = {}
        
        if feature_types is None:
            feature_types = {'feature_1': 'numerical', 'feature_2': 'numerical'}
        
        for feat_name, feat_type in feature_types.items():
            if feat_type == 'numerical':
                # Get distribution or use default normal
                dist = distributions.get(feat_name, {}) if distributions else {}
                dist_type = dist.get('type', 'normal')
                
                if dist_type == 'normal':
                    mean = dist.get('mean', 0)
                    std = dist.get('std', 1)
                    data[feat_name] = np.random.normal(mean, std, n_samples)
                elif dist_type == 'uniform':
                    low = dist.get('low', -1)
                    high = dist.get('high', 1)
                    data[feat_name] = np.random.uniform(low, high, n_samples)
                else:
                    data[feat_name] = np.random.randn(n_samples)
                    
            elif feat_type == 'categorical':
                categories = dist.get('categories', ['A', 'B', 'C']) if distributions else ['A', 'B']
                probs = dist.get('probs', None) if distributions else None
                data[feat_name] = np.random.choice(categories, n_samples, p=probs)
        
        return pd.DataFrame(data)
    
    @staticmethod
    def add_noise(df, noise_level=0.05, columns=None):
        """Add Gaussian noise to numerical columns"""
        df_noisy = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            noise = np.random.normal(0, noise_level * df[col].std(), len(df))
            df_noisy[col] = df[col] + noise
        
        return df_noisy
