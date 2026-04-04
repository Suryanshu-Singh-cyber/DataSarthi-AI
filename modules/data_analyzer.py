import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def data_quality_score(df):
    missing = df.isnull().sum().sum()
    total = df.size
    
    missing_ratio = missing / total
    
    if missing_ratio < 0.05:
        return "High Quality ✅"
    elif missing_ratio < 0.2:
        return "Medium Quality ⚠️"
    else:
        return "Low Quality ❌"

def plot_correlation(df):
    plt.figure(figsize=(6,4))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
    return plt
def analyze_data(df):
    report = {}
    
    report['shape'] = df.shape
    report['missing'] = df.isnull().sum().to_dict()
    report['describe'] = df.describe().to_dict()
    
    return report
