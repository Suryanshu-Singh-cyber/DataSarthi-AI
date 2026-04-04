import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

class ModelRecommender:
    
    @staticmethod
    def recommend(dataset_properties):
        """
        Recommend best ML model based on dataset properties
        
        Parameters:
        dataset_properties: dict with keys:
            - n_samples: number of samples
            - n_features: number of features
            - n_classes: number of classes (for classification)
            - task_type: 'classification', 'regression', 'clustering'
            - linearity: 'linear' or 'nonlinear' (estimated)
        """
        task_type = dataset_properties.get('task_type', 'classification')
        n_samples = dataset_properties.get('n_samples', 1000)
        n_features = dataset_properties.get('n_features', 10)
        n_classes = dataset_properties.get('n_classes', 2)
        linearity = dataset_properties.get('linearity', 'unknown')
        
        recommendations = []
        
        if task_type == 'classification':
            # Rule-based recommendations
            if linearity == 'linear' or n_features > 50:
                recommendations.append({
                    'model': 'Logistic Regression',
                    'reason': 'Good for linear/high-dimensional data',
                    'priority': 1
                })
            
            if n_samples < 1000:
                recommendations.append({
                    'model': 'Naive Bayes',
                    'reason': 'Works well with small datasets',
                    'priority': 2
                })
            
            if linearity == 'nonlinear' or n_samples > 1000:
                recommendations.append({
                    'model': 'Random Forest',
                    'reason': 'Handles non-linear patterns well',
                    'priority': 1
                })
            
            if n_features < 20 and n_samples < 10000:
                recommendations.append({
                    'model': 'SVM',
                    'reason': 'Effective for complex boundaries',
                    'priority': 2
                })
            
            recommendations.append({
                'model': 'Decision Tree',
                'reason': 'Interpretable and easy to understand',
                'priority': 3
            })
            
        elif task_type == 'clustering':
            recommendations.append({
                'model': 'K-Means',
                'reason': 'Standard clustering algorithm',
                'priority': 1
            })
        
        elif task_type == 'regression':
            recommendations.append({
                'model': 'Linear Regression',
                'reason': 'Baseline for regression tasks',
                'priority': 1
            })
        
        # Sort by priority and remove duplicates
        seen = set()
        unique_recs = []
        for rec in sorted(recommendations, key=lambda x: x['priority']):
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
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'K-Means': KMeans(n_clusters=3, random_state=42)
        }
        
        models = {}
        for name in model_names:
            if name in model_map:
                models[name] = model_map[name]
        
        return models
