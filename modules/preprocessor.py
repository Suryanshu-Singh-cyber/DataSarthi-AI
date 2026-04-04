import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer_num = SimpleImputer(strategy='median')
        self.imputer_cat = SimpleImputer(strategy='most_frequent')
    
    def handle_missing_values(self, strategy_num='median', strategy_cat='most_frequent'):
        """Handle missing values in dataset"""
        # Handle numeric columns
        if len(self.numeric_cols) > 0:
            self.imputer_num.set_params(strategy=strategy_num)
            self.df[self.numeric_cols] = self.imputer_num.fit_transform(self.df[self.numeric_cols])
        
        # Handle categorical columns
        if len(self.categorical_cols) > 0:
            self.imputer_cat.set_params(strategy=strategy_cat)
            self.df[self.categorical_cols] = self.imputer_cat.fit_transform(self.df[self.categorical_cols])
        
        return self.df
    
    def encode_categorical(self, method='label'):
        """Encode categorical variables"""
        df_encoded = self.df.copy()
        
        for col in self.categorical_cols:
            if method == 'label':
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
            elif method == 'onehot':
                # For one-hot, we'll return separately
                pass
        
        return df_encoded
    
    def scale_features(self, columns=None):
        """Scale numerical features"""
        if columns is None:
            columns = self.numeric_cols
        
        if len(columns) > 0:
            self.df[columns] = self.scaler.fit_transform(self.df[columns])
        
        return self.df
    
    def remove_outliers(self, method='iqr', threshold=1.5):
        """Remove outliers from dataset"""
        df_clean = self.df.copy()
        
        for col in self.numeric_cols:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        return df_clean
    
    def full_preprocessing_pipeline(self, target_col=None, scale=True, encode=True, 
                                   handle_missing=True, remove_outliers_flag=False):
        """Run complete preprocessing pipeline"""
        df_processed = self.df.copy()
        
        # Separate target if specified
        y = None
        if target_col and target_col in df_processed.columns:
            y = df_processed[target_col]
            df_processed = df_processed.drop(columns=[target_col])
        
        # Handle missing values
        if handle_missing:
            df_processed = DataPreprocessor(df_processed).handle_missing_values()
        
        # Remove outliers
        if remove_outliers_flag:
            df_processed = DataPreprocessor(df_processed).remove_outliers()
        
        # Encode categorical
        if encode:
            preprocessor = DataPreprocessor(df_processed)
            df_processed = preprocessor.encode_categorical()
        
        # Scale features
        if scale:
            preprocessor = DataPreprocessor(df_processed)
            df_processed = preprocessor.scale_features()
        
        # Add target back
        if y is not None:
            df_processed[target_col] = y.values if hasattr(y, 'values') else y
        
        return df_processed
