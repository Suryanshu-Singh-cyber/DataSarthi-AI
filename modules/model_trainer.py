# modules/model_trainer.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, mean_squared_error, r2_score, mean_absolute_error)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

class ModelTrainer:
    
    def __init__(self, df, target_col):
        """
        Initialize the model trainer
        
        Parameters:
        df: pandas DataFrame with features and target
        target_col: name of the target column
        """
        self.df = df.copy()
        self.target_col = target_col
        
        # Separate features and target
        if target_col in df.columns:
            self.X = df.drop(columns=[target_col])
            self.y = df[target_col]
        else:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        # Check if classification or regression
        if self.y.dtype == 'object' or len(self.y.unique()) < 20:
            self.task_type = 'classification'
        else:
            self.task_type = 'regression'
        
        # Handle categorical features
        self.X = pd.get_dummies(self.X)
        
        # Encode target if needed
        if self.y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            self.y = self.label_encoder.fit_transform(self.y)
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
    def prepare_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        return train_test_split(self.X_scaled, self.y, test_size=test_size, random_state=random_state)
    
    def apply_pca(self, n_components=None, variance_threshold=0.95):
        """Apply PCA for dimensionality reduction"""
        if n_components is None:
            pca_temp = PCA()
            pca_temp.fit(self.X_scaled)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_threshold) + 1
        
        pca = PCA(n_components=min(n_components, self.X_scaled.shape[1]))
        X_pca = pca.fit_transform(self.X_scaled)
        
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
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2_score': r2_score(y_test, y_pred)
            }
        
        return {
            'model': model,
            'predictions': y_pred,
            'metrics': metrics
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
                # Initialize model if None was passed
                if model is None:
                    model = self._get_default_model(model_name)
                
                if model is not None:
                    result = self.train_model(model, X_train, X_test, y_train, y_test)
                    results[model_name] = result
                else:
                    results[model_name] = {'error': f"Model {model_name} not recognized"}
            except Exception as e:
                results[model_name] = {'error': str(e)}
        
        return results
    
    def _get_default_model(self, model_name):
        """Get default model instance by name"""
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Naive Bayes': GaussianNB(),
            'SVM': SVC(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        return models.get(model_name)
    
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


# ========== BACKWARD COMPATIBILITY FUNCTIONS ==========

def train_models(X, y, models=None):
    """
    Simple function interface for backward compatibility
    
    Parameters:
    X: feature matrix (numpy array or DataFrame)
    y: target values
    models: dict of model name to model instance (or None for defaults)
    """
    # Create a temporary dataframe
    if isinstance(X, pd.DataFrame):
        temp_df = X.copy()
    else:
        temp_df = pd.DataFrame(X)
    
    temp_df['__target__'] = y
    
    # Create trainer
    trainer = ModelTrainer(temp_df, '__target__')
    
    # Get models to train
    if models is None:
        models_to_train = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
    else:
        models_to_train = {}
        for name, model in models.items():
            if model is None:
                model = trainer._get_default_model(name)
            if model is not None:
                models_to_train[name] = model
    
    # Train models
    results = trainer.train_multiple_models(models_to_train)
    
    return results

def apply_pca(X, n_components=None):
    """Apply PCA for dimensionality reduction"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if n_components is None:
        n_components = min(X_scaled.shape[1], 10)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca
