import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

# Import modules
from modules.dataset_finder import DatasetFinder
from modules.synthetic_generator import SyntheticDataGenerator
from modules.dataset_analyzer import DatasetAnalyzer
from modules.system_checker import SystemCompatibilityChecker
from modules.preprocessor import DataPreprocessor
from modules.model_recommender import ModelRecommender
from modules.model_trainer import ModelTrainer

# Page configuration
st.set_page_config(
    page_title="DataSarthi AI - Intelligent Dataset Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E4057;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #6B7B8F;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F0F2F6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .success-text {
        color: #28A745;
        font-weight: bold;
    }
    .warning-text {
        color: #FFC107;
        font-weight: bold;
    }
    .danger-text {
        color: #DC3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'training_results' not in st.session_state:
    st.session_state.training_results = None

# Title
st.markdown('<h1 class="main-header">🤖 DataSarthi AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;">An Intelligent Dataset Assistant for Discovery, Generation, and ML Optimization</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.markdown("## Navigation")
    
    app_mode = st.radio(
        "Select Mode",
        ["📊 Dataset Finder", "✨ Synthetic Generator", "🔍 Dataset Analyzer", 
         "💻 System Checker", "🤖 Model Trainer", "📈 Dashboard"]
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info("DataSarthi AI helps you discover, generate, analyze, and train ML models on datasets intelligently.")

# Dataset Finder Mode
if app_mode == "📊 Dataset Finder":
    st.markdown('<h2 class="sub-header">🔍 Dataset Finder</h2>', unsafe_allow_html=True)
    
    finder = DatasetFinder()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_term = st.text_input("Search for datasets", placeholder="e.g., iris, cancer, wine...")
    
    with col2:
        st.write("")
        st.write("")
        if st.button("🔄 Show All", use_container_width=True):
            search_term = ""
    
    # Display datasets
    results = finder.search(search_term)
    
    st.markdown(f"### Found {len(results)} datasets")
    
    for idx, row in results.iterrows():
        with st.expander(f"📁 {row['name']}"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Type", row['type'])
            with col2:
                st.metric("Samples", row['samples'])
            with col3:
                st.metric("Features", row['features'])
            with col4:
                st.metric("Source", row['source'])
            
            st.write(f"**Description:** {row['description']}")
            
            if st.button(f"Load {row['name']}", key=f"load_{idx}"):
                # Load dataset from sklearn
                if row['name'] == 'Iris':
                    data = load_iris()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df['target'] = data.target
                    df['target_name'] = data.target_names[data.target]
                elif row['name'] == 'Breast Cancer':
                    data = load_breast_cancer()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df['target'] = data.target
                elif row['name'] == 'Wine':
                    data = load_wine()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df['target'] = data.target
                else:
                    st.warning(f"Dataset {row['name']} loading requires additional configuration")
                    df = None
                
                if df is not None:
                    st.session_state.df = df
                    st.success(f"✅ Loaded {row['name']} dataset!")
                    st.dataframe(df.head())

# Synthetic Generator Mode
elif app_mode == "✨ Synthetic Generator":
    st.markdown('<h2 class="sub-header">✨ Synthetic Data Generator</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Classification", "Regression", "Custom"])
    
    with tab1:
        st.markdown("### Generate Classification Dataset")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            n_samples = st.number_input("Samples", min_value=100, max_value=10000, value=1000)
        with col2:
            n_features = st.number_input("Features", min_value=2, max_value=50, value=10)
        with col3:
            n_classes = st.number_input("Classes", min_value=2, max_value=10, value=2)
        
        if st.button("Generate Classification Data", type="primary"):
            df = SyntheticDataGenerator.generate_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes
            )
            st.session_state.df = df
            st.success(f"✅ Generated {n_samples} samples with {n_features} features")
            st.dataframe(df.head())
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "synthetic_classification.csv", "text/csv")
    
    with tab2:
        st.markdown("### Generate Regression Dataset")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            n_samples = st.number_input("Samples (Reg)", min_value=100, max_value=10000, value=1000)
        with col2:
            n_features = st.number_input("Features (Reg)", min_value=2, max_value=50, value=10)
        with col3:
            noise = st.slider("Noise Level", 0.0, 1.0, 0.1)
        
        if st.button("Generate Regression Data", type="primary"):
            df = SyntheticDataGenerator.generate_regression(
                n_samples=n_samples,
                n_features=n_features,
                noise=noise
            )
            st.session_state.df = df
            st.success(f"✅ Generated {n_samples} samples with {n_features} features")
            st.dataframe(df.head())
    
    with tab3:
        st.markdown("### Custom Dataset Generation")
        
        n_samples_custom = st.number_input("Number of Samples", min_value=10, max_value=5000, value=500)
        
        st.markdown("#### Feature Configuration")
        num_features = st.number_input("Number of Numerical Features", min_value=1, max_value=20, value=2)
        
        feature_types = {}
        for i in range(num_features):
            feature_types[f"feature_{i+1}"] = "numerical"
        
        if st.button("Generate Custom Data"):
            df = SyntheticDataGenerator.generate_custom(
                n_samples=n_samples_custom,
                feature_types=feature_types
            )
            st.session_state.df = df
            st.success("✅ Generated custom dataset!")
            st.dataframe(df.head())

# Dataset Analyzer Mode
elif app_mode == "🔍 Dataset Analyzer":
    st.markdown('<h2 class="sub-header">📊 Dataset Analyzer</h2>', unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        analyzer = DatasetAnalyzer(st.session_state.df)
        results = analyzer.full_analysis_report()
        
        # Display basic info
        col1, col2, col3, col4 = st.columns(4)
        basic = results['basic_info']
        
        with col1:
            st.metric("Rows", basic['rows'])
        with col2:
            st.metric("Columns", basic['columns'])
        with col3:
            st.metric("Missing Values", basic['missing_values'])
