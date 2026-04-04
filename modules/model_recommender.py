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
                    'reason': 'Good for linear/high-dimensional data, fast training',
                    'priority': 1,
                    'pros': ['Fast training', 'Good for high dimensions', 'Interpretable'],
                    'cons': ['Assumes linearity', 'May underfit complex patterns']
                })
            
            if n_samples < 1000:
                recommendations.append({
                    'model': 'Naive Bayes',
                    'reason': 'Works well with small datasets, very fast',
                    'priority': 2,
                    'pros': ['Very fast', 'Works with small data', 'Handles missing values'],
                    'cons': ['Assumes feature independence', 'May be too simplistic']
                })
            
            if linearity == 'nonlinear' or n_samples > 1000:
                recommendations.append({
                    'model': 'Random Forest',
                    'reason': 'Handles non-linear patterns well, robust to overfitting',
                    'priority': 1,
                    'pros': ['Handles non-linearity', 'Feature importance', 'Robust to outliers'],
                    'cons': ['Slower training', 'Less interpretable', 'More memory']
                })
            
            if n_features < 20 and n_samples < 10000:
                recommendations.append({
                    'model': 'SVM',
                    'reason': 'Effective for complex boundaries with kernel trick',
                    'priority': 2,
                    'pros': ['Effective in high dimensions', 'Memory efficient', 'Versatile kernels'],
                    'cons': ['Slow on large datasets', 'Needs feature scaling', 'Hard to interpret']
                })
            
            recommendations.append({
                'model': 'Decision Tree',
                'reason': 'Interpretable and easy to understand, no scaling needed',
                'priority': 3,
                'pros': ['Highly interpretable', 'No scaling needed', 'Handles mixed data'],
                'cons': ['Prone to overfitting', 'Can be unstable', 'May create complex trees']
            })
            
        elif task_type == 'clustering':
            recommendations.append({
                'model': 'K-Means',
                'reason': 'Standard clustering algorithm, fast and efficient',
                'priority': 1,
                'pros': ['Fast', 'Scalable', 'Easy to implement'],
                'cons': ['Need to specify K', 'Assumes spherical clusters', 'Sensitive to initialization']
            })
        
        elif task_type == 'regression':
            recommendations.append({
                'model': 'Linear Regression',
                'reason': 'Baseline for regression tasks, highly interpretable',
                'priority': 1,
                'pros': ['Simple', 'Interpretable', 'Fast training'],
                'cons': ['Assumes linearity', 'Sensitive to outliers', 'May underfit']
            })
            recommendations.append({
                'model': 'Random Forest Regressor',
                'reason': 'Handles non-linear relationships well',
                'priority': 2,
                'pros': ['Handles non-linearity', 'Feature importance', 'Robust'],
                'cons': ['Slower', 'Less interpretable', 'More memory']
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
            'SVM': SVC(random_state=42, probability=True),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
            'K-Means': KMeans(n_clusters=3, random_state=42)
        }
        
        models = {}
        for name in model_names:
            if name in model_map:
                models[name] = model_map[name]
        
        return models
    
    @staticmethod
    def get_model_description(model_name):
        """Get detailed description of a model"""
        descriptions = {
            'Logistic Regression': {
                'best_for': 'Binary classification, linearly separable data',
                'pros': ['Fast training', 'Interpretable coefficients', 'Probability outputs'],
                'cons': ['Assumes linear relationship', 'May underfit complex data']
            },
            'Naive Bayes': {
                'best_for': 'Text classification, small datasets',
                'pros': ['Very fast', 'Works with high dimensions', 'Handles missing data'],
                'cons': ['Assumes feature independence', 'Probability estimates can be biased']
            },
            'SVM': {
                'best_for': 'Complex decision boundaries, medium datasets',
                'pros': ['Effective in high dimensions', 'Memory efficient', 'Versatile kernels'],
                'cons': ['Slow on large datasets', 'Needs careful tuning', 'Less interpretable']
            },
            'Decision Tree': {
                'best_for': 'Interpretable models, feature importance',
                'pros': ['Highly interpretable', 'No scaling needed', 'Handles mixed types'],
                'cons': ['Prone to overfitting', 'Can be unstable', 'May create complex trees']
            },
            'Random Forest': {
                'best_for': 'Complex patterns, feature importance, robust predictions',
                'pros': ['Reduces overfitting', 'Feature importance', 'Handles non-linearity'],
                'cons': ['Slower training', 'Less interpretable', 'More memory usage']
            }
        }
        return descriptions.get(model_name, {})

# Backward compatibility function
def recommend_model(dataset_info):
    """Simple function interface for backward compatibility"""
    recommendations = ModelRecommender.recommend(dataset_info)
    if recommendations:
        return {
            'model': recommendations[0]['model'],
            'reason': recommendations[0]['reason']
        }
    return {'model': 'Random Forest', 'reason': 'Default recommendation for most datasets'}
