import pandas as pd
import numpy as np
from scipy import stats

class DatasetAnalyzer:
    
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    def basic_info(self):
        """Get basic dataset information"""
        return {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'missing_percentage': (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100,
            'duplicate_rows': self.df.duplicated().sum(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
    
    def missing_value_analysis(self):
        """Analyze missing values in dataset"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Missing_Percentage': missing_pct.values
        })
        
        return missing_df[missing_df['Missing_Count'] > 0]
    
    def data_types_analysis(self):
        """Analyze data types"""
        dtype_counts = self.df.dtypes.value_counts()
        return {
            'dtype_summary': dtype_counts.to_dict(),
            'numeric_columns': list(self.numeric_cols),
            'categorical_columns': list(self.categorical_cols)
        }
    
    def statistical_summary(self):
        """Get statistical summary for numeric columns"""
        if len(self.numeric_cols) > 0:
            return self.df[self.numeric_cols].describe().to_dict()
        return {}
    
    def correlation_matrix(self):
        """Calculate correlation matrix for numeric columns"""
        if len(self.numeric_cols) > 1:
            return self.df[self.numeric_cols].corr()
        return pd.DataFrame()
    
    def class_imbalance_detection(self):
        """Check for class imbalance in target column"""
        # Try to find target column (common names)
        target_candidates = ['target', 'class', 'label', 'y']
        target_col = None
        
        for col in target_candidates:
            if col in self.df.columns:
                target_col = col
                break
        
        if target_col is None and len(self.categorical_cols) > 0:
            target_col = self.categorical_cols[0]
        
        if target_col:
            class_counts = self.df[target_col].value_counts()
            class_pct = (class_counts / len(self.df)) * 100
            
            imbalance_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else float('inf')
            
            return {
                'target_column': target_col,
                'class_counts': class_counts.to_dict(),
                'class_percentages': class_pct.to_dict(),
                'imbalance_ratio': imbalance_ratio,
                'is_imbalanced': imbalance_ratio > 3
            }
        return None
    
    def outlier_detection(self, method='iqr', threshold=1.5):
        """Detect outliers using IQR or Z-score method"""
        outliers_summary = {}
        
        for col in self.numeric_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outliers = self.df[z_scores > threshold]
            
            outliers_summary[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(self.df)) * 100
            }
        
        return outliers_summary
    
    def full_analysis_report(self):
        """Generate complete analysis report"""
        return {
            'basic_info': self.basic_info(),
            'missing_values': self.missing_value_analysis().to_dict('records') if len(self.missing_value_analysis()) > 0 else [],
            'data_types': self.data_types_analysis(),
            'statistics': self.statistical_summary(),
            'class_imbalance': self.class_imbalance_detection(),
            'outliers': self.outlier_detection()
        }
