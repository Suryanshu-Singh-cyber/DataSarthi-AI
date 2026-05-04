# modules/recommender.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

class ModelRecommender:
    
    @staticmethod
    def recommend(dataset_properties):
        """Recommend best ML model based on dataset properties"""
        task_type = dataset_properties.get('task_type', 'classification')
        n_samples = dataset_properties.get('n_samples', 1000)
        n_features = dataset_properties.get('n_features', 10)
        
        recommendations = []
        
        if task_type == 'classification':
            if n_features > 50:
                recommendations.append({
                    'model': 'Logistic Regression',
                    'reason': 'Good for high-dimensional data',
                    'priority': 1
                })
            
            if n_samples < 1000:
                recommendations.append({
                    'model': 'Naive Bayes',
                    'reason': 'Works well with small datasets',
                    'priority': 2
                })
            
            if n_samples > 1000:
                recommendations.append({
                    'model': 'Random Forest',
                    'reason': 'Handles complex patterns well',
                    'priority': 1
                })
            
            recommendations.append({
                'model': 'Decision Tree',
                'reason': 'Interpretable and easy to understand',
                'priority': 3
            })
        
        elif task_type == 'regression':
            recommendations.append({
                'model': 'Linear Regression',
                'reason': 'Baseline for regression tasks',
                'priority': 1
            })
        
        # Remove duplicates
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec['model'] not in seen:
                seen.add(rec['model'])
                unique_recs.append(rec)
        
        return unique_recs
    
    @staticmethod
    def get_model_instances(model_names):
        """Get actual model instances for training"""
        model_map = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Naive Bayes': GaussianNB(),
            'SVM': SVC(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        models = {}
        for name in model_names:
            if name in model_map:
                models[name] = model_map[name]
        
        return models


def recommend_model(dataset_info):
    """Simple function interface for backward compatibility"""
    recommendations = ModelRecommender.recommend(dataset_info)
    if recommendations:
        return {
            'model': recommendations[0]['model'],
            'reason': recommendations[0]['reason']
        }
    return {'model': 'Random Forest', 'reason': 'Default recommendation'}
