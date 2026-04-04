import pandas as pd
import numpy as np

class DatasetFinder:
    def __init__(self):
        self.catalog = self._create_catalog()
    
    def _create_catalog(self):
        """Create predefined dataset catalog"""
        return pd.DataFrame({
            'name': ['Iris', 'Breast Cancer', 'Wine', 'Diabetes', 'Digits'],
            'description': [
                'Classic flower classification dataset (3 classes, 4 features)',
                'Breast cancer diagnosis dataset (2 classes, 30 features)',
                'Wine classification dataset (3 classes, 13 features)',
                'Diabetes progression dataset (Regression)',
                'Handwritten digits recognition (10 classes, 64 features)'
            ],
            'type': ['Classification', 'Classification', 'Classification', 'Regression', 'Classification'],
            'samples': [150, 569, 178, 442, 1797],
            'features': [4, 30, 13, 10, 64],
            'source': ['sklearn', 'sklearn', 'sklearn', 'sklearn', 'sklearn']
        })
    
    def search(self, keyword):
        """Search datasets by keyword"""
        if not keyword:
            return self.catalog
        
        mask = (
            self.catalog['name'].str.contains(keyword, case=False) |
            self.catalog['description'].str.contains(keyword, case=False)
        )
        return self.catalog[mask]
    
    def get_dataset_info(self, dataset_name):
        """Get detailed info about a dataset"""
        result = self.catalog[self.catalog['name'].str.contains(dataset_name, case=False)]
        if not result.empty:
            return result.iloc[0].to_dict()
        return None
