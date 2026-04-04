import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
