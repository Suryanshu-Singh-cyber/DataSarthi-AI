import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class ModelTrainer:
    
    def __init__(self, df, target_col):
        self.df = df
        self.target_col = target_col
        self.X = df.drop(columns=[target_col])
        self.y = df[target_col]
        
        # Check if classification or regression
        if self.y.dtype == 'object' or len(self.y.unique()) < 20:
            self.task_type = 'classification'
        else:
            self.task_type = 'regression'
        
    def prepare_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
    
    def apply_pca(self, n_components=None, variance_threshold=0.95):
        """Apply PCA for dimensionality reduction"""
        if n_components is None:
            # Determine number of components to explain variance_threshold variance
            pca_temp = PCA()
            pca_temp.fit(self.X)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_threshold) + 1
        
        pca = PCA(n_components=min(n_components, self.X.shape[1]))
        X_pca = pca.fit_transform(self.X)
        
        return X_pca, pca
    
    def train_model(self, model, X_train, X_test, y_train, y_test):
        """Train a single model and return metrics"""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if self.task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
        else:
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2_score': r2_score(y_test, y_pred)
            }
        
        return {
            'model': model,
            'predictions': y_pred,
            'metrics': metrics,
            'confusion_matrix': confusion_matrix(y_test, y_pred) if self.task_type == 'classification' else None
        }
    
    def train_multiple_models(self, models_dict, use_pca=False, pca_components=None):
        """Train multiple models and compare performance"""
        results = {}
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Apply PCA if requested
        if use_pca and X_train.shape[1] > 2:
            pca = PCA(n_components=min(pca_components or X_train.shape[1], X_train.shape[1]))
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            results['pca_info'] = {
                'n_components': X_train.shape[1],
                'explained_variance': pca.explained_variance_ratio_.sum()
            }
        
        # Train each model
        for model_name, model in models_dict.items():
            try:
                result = self.train_model(model, X_train, X_test, y_train, y_test)
                results[model_name] = result
            except Exception as e:
                results[model_name] = {'error': str(e)}
        
        return results
    
    def get_best_model(self, results, metric='accuracy'):
        """Get the best performing model"""
        best_model_name = None
        best_score = -1
        
        for model_name, result in results.items():
            if isinstance(result, dict) and 'metrics' in result and metric in result['metrics']:
                score = result['metrics'][metric]
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
        
        if best_model_name:
            return {
                'best_model': best_model_name,
                'best_score': best_score,
                'metric': metric,
                'full_results': {name: res.get('metrics', {}) for name, res in results.items() 
                               if isinstance(res, dict) and 'metrics' in res}
            }
        return None

# Backward compatibility functions
def train_models(X, y, models=None):
    """Simple function interface for backward compatibility"""
    if models is None or not models:
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100)
        }
    
    # Create a temporary dataframe
    temp_df = pd.DataFrame(X)
    if len(temp_df.columns) != X.shape[1]:
        # X is already numpy array
        pass
    
    trainer = ModelTrainer(pd.DataFrame(X), 'temp_target')
    trainer.y = y
    trainer.X = X
    trainer.task_type = 'classification'
    
    return trainer.train_multiple_models(models)

def apply_pca(X, n_components=None):
    """Apply PCA for dimensionality reduction"""
    pca = PCA(n_components=n_components if n_components else min(X.shape[1], 10))
    X_pca = pca.fit_transform(X)
    return X_pca, pca
