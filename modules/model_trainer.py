# modules/model_trainer.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class ModelTrainer:
    
    def __init__(self, df, target_col):
        """Initialize the model trainer"""
        self.df = df.copy()
        self.target_col = target_col
        self.results = {}
        
        # Separate features and target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found!")
        
        self.X = df.drop(columns=[target_col])
        self.y = df[target_col].copy()
        
        # Check task type
        if self.y.dtype == 'object' or len(self.y.unique()) < 20:
            self.task_type = 'classification'
        else:
            self.task_type = 'regression'
        
        # Preprocess
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess features and target"""
        # Handle categorical features
        self.X = pd.get_dummies(self.X)
        
        # Encode target if classification and categorical
        if self.task_type == 'classification' and self.y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            self.y = self.label_encoder.fit_transform(self.y)
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
    
    def train_multiple_models(self, model_names, use_pca=False, pca_components=None, test_size=0.2):
        """Train multiple models and return comparison results"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=42, stratify=self.y if self.task_type == 'classification' else None
        )
        
        # Apply PCA if requested
        if use_pca and X_train.shape[1] > 2:
            n_components = min(pca_components or min(10, X_train.shape[1]), X_train.shape[1])
            pca = PCA(n_components=n_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            self.pca_applied = True
            self.pca_info = {
                'n_components': n_components,
                'explained_variance': pca.explained_variance_ratio_.sum()
            }
        else:
            self.pca_applied = False
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Naive Bayes': GaussianNB(),
            'SVM': SVC(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        # Train only selected models
        results = {}
        for name in model_names:
            if name in models:
                try:
                    model = models[name]
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    if self.task_type == 'classification':
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        
                        results[name] = {
                            'accuracy': round(accuracy, 4),
                            'precision': round(precision, 4),
                            'recall': round(recall, 4),
                            'f1_score': round(f1, 4)
                        }
                    else:
                        from sklearn.metrics import mean_squared_error, r2_score
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        results[name] = {
                            'mse': round(mse, 4),
                            'rmse': round(np.sqrt(mse), 4),
                            'r2_score': round(r2, 4)
                        }
                        
                except Exception as e:
                    results[name] = {'error': str(e)}
        
        self.results = results
        return results
    
    def get_best_model(self, metric='accuracy'):
        """Get the best performing model"""
        if not self.results:
            return None
        
        best_model = None
        best_score = -1
        
        for model_name, metrics in self.results.items():
            if metric in metrics and isinstance(metrics[metric], (int, float)):
                if metrics[metric] > best_score:
                    best_score = metrics[metric]
                    best_model = model_name
        
        if best_model:
            return {
                'best_model': best_model,
                'best_score': best_score,
                'metric': metric,
                'all_results': self.results
            }
        return None
    
    def get_results_dataframe(self):
        """Get results as pandas DataFrame"""
        if not self.results:
            return pd.DataFrame()
        
        df_results = pd.DataFrame(self.results).T
        return df_results


# Simple function for backward compatibility
def train_models(X, y, models_dict):
    """Simple function interface"""
    # Create temporary dataframe
    temp_df = pd.DataFrame(X)
    temp_df['__target__'] = y
    
    trainer = ModelTrainer(temp_df, '__target__')
    return trainer.train_multiple_models(list(models_dict.keys()))
