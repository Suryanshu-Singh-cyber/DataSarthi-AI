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
        with col4:
            st.metric("Memory Usage", f"{basic['memory_usage_mb']:.2f} MB")
        
        # Missing values
        if results['missing_values']:
            st.markdown("### Missing Values Analysis")
            st.dataframe(pd.DataFrame(results['missing_values']))
        
        # Statistical summary
        if results['statistics']:
            st.markdown("### Statistical Summary")
            st.dataframe(pd.DataFrame(results['statistics']).T)
        
        # Correlation matrix
        corr_matrix = analyzer.correlation_matrix()
        if not corr_matrix.empty:
            st.markdown("### Correlation Matrix")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        
        # Class imbalance
        if results['class_imbalance']:
            st.markdown("### Class Imbalance Analysis")
            imb = results['class_imbalance']
            st.write(f"**Target Column:** {imb['target_column']}")
            
            if imb['is_imbalanced']:
                st.warning(f"⚠️ Dataset is imbalanced! Imbalance Ratio: {imb['imbalance_ratio']:.2f}")
            else:
                st.success(f"✅ Dataset is balanced. Imbalance Ratio: {imb['imbalance_ratio']:.2f}")
            
            # Class distribution chart
            fig = px.bar(
                x=list(imb['class_counts'].keys()),
                y=list(imb['class_counts'].values()),
                title="Class Distribution"
            )
            st.plotly_chart(fig)
        
        # Outliers
        st.markdown("### Outlier Analysis")
        outliers = results['outliers']
        outlier_df = pd.DataFrame(outliers).T
        st.dataframe(outlier_df)
        
    else:
        st.warning("⚠️ Please load or generate a dataset first!")

# System Checker Mode
elif app_mode == "💻 System Checker":
    st.markdown('<h2 class="sub-header">💻 System Compatibility Checker</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        ram_gb = st.number_input("RAM (GB)", min_value=1, max_value=128, value=8)
        cpu_cores = st.number_input("CPU Cores", min_value=1, max_value=32, value=4)
    
    with col2:
        if st.session_state.df is not None:
            rows = len(st.session_state.df)
            cols = len(st.session_state.df.columns)
            st.metric("Current Dataset", f"{rows} rows × {cols} cols")
        else:
            rows = st.number_input("Dataset Rows", min_value=100, max_value=1000000, value=10000)
            cols = st.number_input("Dataset Columns", min_value=1, max_value=500, value=10)
    
    if st.button("Check Compatibility", type="primary"):
        result = SystemCompatibilityChecker.check_compatibility(ram_gb, cpu_cores, rows, cols)
        
        st.markdown("### Compatibility Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(result['memory_message'])
        with col2:
            st.info(result['cpu_message'])
        
        if result['compatible']:
            st.success(f"✅ Dataset is compatible with your system!")
        else:
            st.error(f"❌ Dataset may not be compatible!")
        
        st.markdown(f"**Estimated Memory Usage:** {result['estimated_memory_gb']} GB")
        st.markdown(f"**Available RAM (70%):** {result['available_ram_gb']} GB")
        st.markdown(f"**{result['recommendation']}**")
        
        # Optimizations
        suggestions = SystemCompatibilityChecker.suggest_optimizations(rows, cols, ram_gb)
        if suggestions:
            st.markdown("### Optimization Suggestions")
            for s in suggestions:
                st.write(f"💡 {s}")

# Model Trainer Mode
elif app_mode == "🤖 Model Trainer":
    st.markdown('<h2 class="sub-header">🤖 Model Training & Comparison</h2>', unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        # Select target column
        target_col = st.selectbox("Select Target Column", st.session_state.df.columns)
        
        # Model selection
        st.markdown("### Select Models to Train")
        
        available_models = ['Logistic Regression', 'Naive Bayes', 'SVM', 'Decision Tree', 'Random Forest']
        selected_models = st.multiselect("Choose models", available_models, default=['Random Forest', 'Logistic Regression'])
        
        # Advanced options
        with st.expander("Advanced Options"):
            use_pca = st.checkbox("Apply PCA (Dimensionality Reduction)")
            pca_components = None
            if use_pca:
                pca_components = st.number_input("PCA Components", min_value=2, max_value=50, value=10)
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
        
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models... This may take a moment."):
                # Get model instances
                model_dict = ModelRecommender.get_model_instances(selected_models)
                
                # Train models
                trainer = ModelTrainer(st.session_state.df, target_col)
                results = trainer.train_multiple_models(model_dict, use_pca=use_pca, pca_components=pca_components)
                
                st.session_state.training_results = results
                
                # Display results
                st.markdown("### Training Results")
                
                # Create comparison dataframe
                comparison_data = []
                for model_name, result in results.items():
                    if 'metrics' in result:
                        row = {'Model': model_name}
                        row.update(result['metrics'])
                        comparison_data.append(row)
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df.style.highlight_max(color='lightgreen', subset=['accuracy', 'f1_score']))
                    
                    # Best model
                    best = trainer.get_best_model(results)
                    if best:
                        st.success(f"🏆 **Best Model:** {best['best_model']} with {best['metric']} = {best['best_score']:.4f}")
                    
                    # Visualization
                    st.markdown("### Performance Visualization")
                    fig = go.Figure()
                    
                    for model_name in comparison_df['Model']:
                        model_data = comparison_df[comparison_df['Model'] == model_name]
                        if 'accuracy' in model_data.columns:
                            fig.add_trace(go.Bar(
                                name=model_name,
                                x=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                                y=[
                                    model_data['accuracy'].values[0] if 'accuracy' in model_data.columns else 0,
                                    model_data['precision'].values[0] if 'precision' in model_data.columns else 0,
                                    model_data['recall'].values[0] if 'recall' in model_data.columns else 0,
                                    model_data['f1_score'].values[0] if 'f1_score' in model_data.columns else 0
                                ]
                            ))
                    
                    fig.update_layout(title="Model Performance Comparison", barmode='group')
                    st.plotly_chart(fig)
                
                if use_pca and 'pca_info' in results:
                    st.info(f"📊 PCA applied: Reduced to {results['pca_info']['n_components']} components")
    
    else:
        st.warning("⚠️ Please load or generate a dataset first!")

# Dashboard Mode
elif app_mode == "📈 Dashboard":
    st.markdown('<h2 class="sub-header">📊 Complete Dashboard</h2>', unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        # Dataset summary
        st.markdown("### Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", len(st.session_state.df))
        with col2:
            st.metric("Total Columns", len(st.session_state.df.columns))
        with col3:
            missing = st.session_state.df.isnull().sum().sum()
            st.metric("Missing Values", missing)
        with col4:
            duplicates = st.session_state.df.duplicated().sum()
            st.metric("Duplicate Rows", duplicates)
        
        # Data preview
        st.markdown("### Data Preview")
        st.dataframe(st.session_state.df.head(10))
        
        # Data types
        st.markdown("### Data Types")
        dtype_df = pd.DataFrame({
            'Column': st.session_state.df.dtypes.index,
            'Type': st.session_state.df.dtypes.values
        })
        st.dataframe(dtype_df)
        
        # Numerical columns distribution
        num_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            st.markdown("### Feature Distributions")
            selected_col = st.selectbox("Select feature to visualize", num_cols)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Histogram
            st.session_state.df[selected_col].hist(bins=30, ax=axes[0])
            axes[0].set_title(f"Histogram of {selected_col}")
            axes[0].set_xlabel(selected_col)
            
            # Box plot
            st.session_state.df.boxplot(column=selected_col, ax=axes[1])
            axes[1].set_title(f"Box Plot of {selected_col}")
            
            st.pyplot(fig)
        
        # Recommendations
        st.markdown("### ML Recommendations")
        
        # Estimate dataset properties
        properties = {
            'n_samples': len(st.session_state.df),
            'n_features': len(st.session_state.df.columns),
            'task_type': 'classification',  # Default
            'linearity': 'unknown'
        }
        
        recommendations = ModelRecommender.recommend(properties)
        
        for rec in recommendations[:3]:
            st.info(f"💡 **{rec['model']}** - {rec['reason']}")
        
        # Download report
        st.markdown("### Download Report")
        
        # Create report
        report_buffer = BytesIO()
        with pd.ExcelWriter(report_buffer, engine='openpyxl') as writer:
            st.session_state.df.to_excel(writer, sheet_name='Data', index=False)
            
            # Add summary
            summary_df = pd.DataFrame({
                'Metric': ['Rows', 'Columns', 'Missing Values', 'Duplicate Rows'],
                'Value': [len(st.session_state.df), len(st.session_state.df.columns), 
                         missing, duplicates]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        report_buffer.seek(0)
        st.download_button(
            label="📥 Download Report (Excel)",
            data=report_buffer,
            filename="datasarthi_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    else:
        st.warning("⚠️ Please load or generate a dataset first!")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #6B7B8F;'>🤖 DataSarthi AI - Your Intelligent Dataset Assistant</p>",
    unsafe_allow_html=True
)
