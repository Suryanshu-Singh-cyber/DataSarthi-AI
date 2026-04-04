import pandas as pd

def analyze_data(df):
    report = {}
    
    report['shape'] = df.shape
    report['missing'] = df.isnull().sum().to_dict()
    report['describe'] = df.describe().to_dict()
    
    return report
