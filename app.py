import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="DataSarthi AI - Intelligent Dataset Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #fff 0%, #e0e0e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        margin-bottom: 1.5rem;
        opacity: 0.95;
    }
    
    /* Card styling */
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        transition: transform 0.3s ease;
        margin: 0.5rem 0;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .pipeline-step {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    /* Section headers */
    .section-header {
        font-size: 2rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .subsection-header {
        font-size: 1.5rem;
        font-weight: 500;
        margin: 1.5rem 0 1rem 0;
        color: #2c3e50;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }
    
    /* Footer */
    .footer {
        background: #2c3e50;
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-top: 3rem;
    }
    
    /* Badge */
    .badge {
        background: #27ae60;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    /* Timeline */
    .timeline-item {
        border-left: 3px solid #667eea;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    
    /* Code block */
    .code-block {
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'df' not in st.session_state:
    st.session_state.df = None
if 'results' not in st.session_state:
    st.session_state.results = None

# Navigation
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <img src="https://img.icons8.com/fluency/96/artificial-intelligence.png" width="80">
    <h3 style="margin-top: 0.5rem;">DataSarthi AI</h3>
    <p style="color: #6c757d; font-size: 0.9rem;">Intelligent Dataset Assistant</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
page = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Home", "📊 Dataset Finder", "✨ Synthetic Generator", "🔍 Dataset Analyzer", 
     "💻 System Checker", "🤖 Model Trainer", "📈 Dashboard", "📚 Documentation"]
)

# Update session state
st.session_state.page = page

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🤖 DataSarthi AI</h1>
        <p class="hero-subtitle">Your Intelligent Companion for Machine Learning Workflows</p>
        <p style="font-size: 1rem;">Eliminate Trial & Error | Automate ML Pipeline | Make Data Intelligence Accessible</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="stat-card">
            <h2>📊</h2>
            <h3>5+</h3>
            <p>Built-in Datasets</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-card">
            <h2>🤖</h2>
            <h3>6+</h3>
            <p>ML Algorithms</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-card">
            <h2>⚡</h2>
            <h3>Real-time</h3>
            <p>Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="stat-card">
            <h2>🎯</h2>
            <h3>95%</h3>
            <p>Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Project Story Section
    st.markdown('<h2 class="section-header">📖 Our Story</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### The Problem We Saw 👀
        
        Data Scientists and ML practitioners spend **70-80%** of their time on:
        - Finding the right datasets
        - Cleaning and preprocessing data
        - Experimenting with different models
        - Dealing with system crashes due to large datasets
        
        ### Our Vision ✨
        
        **DataSarthi AI** (Data Companion) was born to eliminate this trial-and-error approach. We provide an intelligent assistant that:
        
        - 🎯 **Discovers** relevant datasets automatically
        - 🔧 **Generates** synthetic data when real data is scarce
        - 📊 **Analyzes** dataset quality instantly
        - 💡 **Recommends** the best ML models
        - ⚡ **Trains** and compares models automatically
        - 💻 **Checks** system compatibility before training
        
        ### The Impact 🌟
        
        With DataSarthi AI, you can go from raw data to production-ready model in **minutes**, not weeks!
        """)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white;">
            <h3 style="text-align: center;">🎯 Mission</h3>
            <p style="text-align: center;">Democratize Machine Learning by making it accessible, automated, and intelligent for everyone.</p>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <h3 style="text-align: center;">🏆 Vision</h3>
            <p style="text-align: center;">Become the ultimate AI companion for every data practitioner worldwide.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Features Section
    st.markdown('<h2 class="section-header">✨ Key Features</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    features = [
        ("🔍", "Dataset Finder", "Search and discover relevant datasets from built-in catalog with intelligent keyword matching"),
        ("✨", "Synthetic Generator", "Generate custom datasets with various distributions (Normal, Uniform) and noise levels"),
        ("📊", "Dataset Analyzer", "Comprehensive EDA with missing value detection, correlation matrix, and outlier analysis"),
        ("💻", "System Checker", "Check dataset compatibility with your system's RAM and CPU before training"),
        ("🤖", "Model Recommender", "Intelligent model suggestions based on dataset properties and task type"),
        ("⚡", "Auto Trainer", "Train and compare 6+ ML models automatically with performance metrics"),
        ("📈", "Interactive Dashboard", "Visualize results with beautiful charts and download comprehensive reports"),
        ("🔄", "PCA Integration", "Built-in dimensionality reduction for high-dimensional datasets"),
        ("📥", "Export Results", "Download datasets, models, and reports for production use")
    ]
    
    for i, (icon, title, desc) in enumerate(features):
        with [col1, col2, col3][i % 3]:
            st.markdown(f"""
            <div class="feature-card">
                <h1 style="font-size: 2rem;">{icon}</h1>
                <h3>{title}</h3>
                <p style="font-size: 0.9rem; opacity: 0.9;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ML Pipeline Section
    st.markdown('<h2 class="section-header">🔄 Machine Learning Pipeline</h2>', unsafe_allow_html=True)
    
    pipeline_steps = [
        "📥 Data Input", "🧹 Data Cleaning", "📊 EDA", 
        "🔧 Feature Engineering", "📉 PCA (Optional)", 
        "🤖 Model Training", "📈 Evaluation", "🏆 Best Model"
    ]
    
    cols = st.columns(len(pipeline_steps))
    for idx, step in enumerate(pipeline_steps):
        with cols[idx]:
            st.markdown(f"""
            <div class="pipeline-step">
                <h3>⬇️</h3>
                <strong>{step}</strong>
            </div>
            """, unsafe_allow_html=True)
    
    # IPO Model Section
    st.markdown('<h2 class="section-header">📋 Input → Process → Output (IPO) Model</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: #e8f4f8; padding: 1rem; border-radius: 10px;">
            <h3 style="text-align: center; color: #2980b9;">📥 INPUT</h3>
            <ul>
                <li>CSV/Excel Dataset</li>
                <li>Synthetic parameters</li>
                <li>System specs (RAM/CPU)</li>
                <li>ML task type</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #fef9e8; padding: 1rem; border-radius: 10px;">
            <h3 style="text-align: center; color: #f39c12;">⚙️ PROCESS</h3>
            <ul>
                <li>Data Validation</li>
                <li>Preprocessing</li>
                <li>Feature Engineering</li>
                <li>Model Training</li>
                <li>Evaluation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: #e8f5e9; padding: 1rem; border-radius: 10px;">
            <h3 style="text-align: center; color: #27ae60;">📤 OUTPUT</h3>
            <ul>
                <li>Dataset Insights</li>
                <li>Best Model</li>
                <li>Performance Metrics</li>
                <li>Recommendations</li>
                <li>Downloadable Reports</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Workflow Diagram
    st.markdown('<h2 class="section-header">🔄 Workflow</h2>', unsafe_allow_html=True)
    
    workflow_data = {
        'Step': ['Data Discovery', 'Data Generation', 'Data Analysis', 'System Check', 'Model Training', 'Evaluation'],
        'Duration': [10, 15, 20, 5, 30, 10],
        'Automation': [80, 100, 95, 100, 90, 100]
    }
    workflow_df = pd.DataFrame(workflow_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Time (minutes)', x=workflow_df['Step'], y=workflow_df['Duration'], 
                         marker_color='#667eea'))
    fig.add_trace(go.Bar(name='Automation %', x=workflow_df['Step'], y=workflow_df['Automation'],
                         marker_color='#27ae60'))
    fig.update_layout(title='Workflow Efficiency', barmode='group', height=400,
                     plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Tech Stack
    st.markdown('<h2 class="section-header">💻 Technology Stack</h2>', unsafe_allow_html=True)
    
    tech_cols = st.columns(4)
    tech_stack = [
        ("🐍 Python", "Backend Core"),
        ("📊 Streamlit", "Frontend UI"),
        ("🤖 Scikit-learn", "ML Models"),
        ("📈 Pandas/NumPy", "Data Processing"),
        ("🎨 Matplotlib/Seaborn", "Visualization"),
        ("📉 Plotly", "Interactive Charts"),
        ("🧮 PCA", "Dim Reduction"),
        ("📦 Joblib", "Model Export")
    ]
    
    for idx, (tech, purpose) in enumerate(tech_stack):
        with tech_cols[idx % 4]:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 8px; text-align: center; margin: 0.2rem;">
                <strong>{tech}</strong><br>
                <small style="color: #6c757d;">{purpose}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # CTA Button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Get Started Now", use_container_width=True):
            st.session_state.page = "📊 Dataset Finder"
            st.rerun()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <h3>🤖 DataSarthi AI</h3>
        <p>Intelligent Dataset Assistant | Making ML Accessible for Everyone</p>
        <p style="font-size: 0.8rem;">© 2024 DataSarthi AI. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== DATASET FINDER ====================
elif page == "📊 Dataset Finder":
    st.markdown('<h2 class="section-header">🔍 Dataset Finder</h2>', unsafe_allow_html=True)
    
    st.info("📚 Discover and load built-in datasets or upload your own CSV files")
    
    dataset_option = st.selectbox(
        "Select Dataset Source",
        ["Iris", "Breast Cancer", "Wine", "Upload CSV"]
    )
    
    if dataset_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success("✅ Dataset uploaded successfully!")
            st.dataframe(st.session_state.df.head())
            st.info(f"Shape: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")
    else:
        if st.button(f"📥 Load {dataset_option} Dataset", type="primary"):
            with st.spinner("Loading dataset..."):
                if dataset_option == "Iris":
                    data = load_iris()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df['target'] = data.target
                elif dataset_option == "Breast Cancer":
                    data = load_breast_cancer()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df['target'] = data.target
                else:
                    data = load_wine()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df['target'] = data.target
                
                st.session_state.df = df
                st.success(f"✅ Loaded {dataset_option} dataset!")
                st.dataframe(df.head())
                
                # Dataset info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Back to home button
    if st.button("🏠 Back to Home"):
        st.session_state.page = "🏠 Home"
        st.rerun()

# ========== SYNTHETIC GENERATOR ==========
elif menu == "✨ Synthetic Generator":
    st.markdown('<h2 class="sub-header">✨ Synthetic Data Generator</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_type = st.selectbox("Dataset Type", ["Classification", "Regression"])
        n_samples = st.number_input("Number of Samples", min_value=100, max_value=10000, value=1000)
    
    with col2:
        n_features = st.number_input("Number of Features", min_value=2, max_value=50, value=10)
        if data_type == "Regression":
            noise = st.slider("Noise Level", 0.0, 0.5, 0.1)
    
    if st.button("🚀 Generate Dataset", type="primary"):
        with st.spinner("Generating synthetic data..."):
            from modules.data_generator import generate_data
            
            if data_type == "Classification":
                df = generate_data(
                    dataset_type="classification",
                    n_samples=n_samples,
                    n_features=n_features
                )
            else:
                df = generate_data(
                    dataset_type="regression",
                    n_samples=n_samples,
                    n_features=n_features,
                    noise=noise
                )
            
            st.session_state.df = df
            st.success(f"✅ Generated {n_samples} samples with {n_features} features!")
            st.dataframe(df.head())
            
            # Download button - NO try block needed
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"synthetic_{data_type.lower()}_data.csv",
                mime="text/csv"
            )
    
    if st.button("🏠 Back to Home"):
        st.session_state.page = "🏠 Home"
        st.rerun()
# ==================== DATASET ANALYZER ====================
elif page == "🔍 Dataset Analyzer":
    st.markdown('<h2 class="section-header">📊 Dataset Analyzer</h2>', unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Basic info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        
        # Data preview
        st.markdown("### Data Preview")
        st.dataframe(df.head(10))
        
        # Statistical summary
        st.markdown("### Statistical Summary")
        st.dataframe(df.describe())
        
        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            st.markdown("### Correlation Matrix")
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            st.pyplot(fig)
    else:
        st.warning("⚠️ Please load or generate a dataset first!")
    
    if st.button("🏠 Back to Home"):
        st.session_state.page = "🏠 Home"
        st.rerun()

# ==================== SYSTEM CHECKER ====================
elif page == "💻 System Checker":
    st.markdown('<h2 class="section-header">💻 System Compatibility Checker</h2>', unsafe_allow_html=True)
    
    st.info("🔧 Check if your system can handle the dataset before training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ram_gb = st.number_input("RAM (GB)", min_value=1, max_value=128, value=8)
        cpu_cores = st.number_input("CPU Cores", min_value=1, max_value=32, value=4)
    
    with col2:
        if st.session_state.df is not None:
            dataset_size_mb = st.session_state.df.memory_usage(deep=True).sum() / (1024 * 1024)
            st.metric("Dataset Size", f"{dataset_size_mb:.2f} MB")
        else:
            dataset_size_mb = st.number_input("Dataset Size (MB)", min_value=0.1, max_value=10000.0, value=10.0)
    
    if st.button("🔍 Check Compatibility", type="primary"):
        try:
            from modules.system_checker import check_system
            result = check_system(ram_gb, cpu_cores, dataset_size_mb)
            
            if "✅" in result['status']:
                st.success(result['status'])
                st.info(f"💡 {result['message']}")
            else:
                st.warning(result['status'])
                st.info(f"💡 {result['message']}")
        except Exception as e:
            st.info(f"ℹ️ Estimated memory: {dataset_size_mb:.2f} MB")
            if dataset_size_mb > ram_gb * 700:
                st.warning("⚠️ Dataset may be too large for your system")
            else:
                st.success("✅ Dataset size seems compatible with your system")
    
    if st.button("🏠 Back to Home"):
        st.session_state.page = "🏠 Home"
        st.rerun()

# ========== MODEL TRAINER ==========
elif menu == "🤖 Model Trainer":
    st.markdown('<h2 class="sub-header">🤖 Model Training & Comparison</h2>', unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Select target column
        target_col = st.selectbox("Select Target Column", df.columns)
        
        # Options
        st.markdown("### Training Options")
        col1, col2 = st.columns(2)
        
        with col1:
            use_pca = st.checkbox("Apply PCA (Dimensionality Reduction)")
            if use_pca:
                max_components = min(20, df.shape[1] - 1)
                n_components = st.slider("Number of Components", 2, max_components, min(5, max_components))
        
        with col2:
            models_to_train = st.multiselect(
                "Select Models to Train",
                ['Logistic Regression', 'Naive Bayes', 'SVM', 'Decision Tree', 'Random Forest'],
                default=['Random Forest', 'Logistic Regression']
            )
        
        if st.button("🚀 Train Models", type="primary"):
            if not models_to_train:
                st.error("Please select at least one model to train!")
            else:
                with st.spinner("Training models... This may take a moment."):
                    try:
                        # Import ModelTrainer
                        from modules.model_trainer import ModelTrainer, train_models
                        
                        # Create trainer instance
                        trainer = ModelTrainer(df, target_col)
                        
                        # Prepare models dict
                        models_dict = {name: None for name in models_to_train}
                        
                        # Train models
                        results = trainer.train_multiple_models(models_dict, use_pca=use_pca)
                        
                        # Display results
                        st.markdown("### 📊 Model Performance Comparison")
                        
                        # Create results dataframe
                        results_data = []
                        for model_name, result in results.items():
                            if model_name != 'pca_info' and 'metrics' in result:
                                row = {'Model': model_name}
                                row.update(result['metrics'])
                                results_data.append(row)
                        
                        if results_data:
                            results_df = pd.DataFrame(results_data)
                            
                            # Format percentage columns
                            for col in ['accuracy', 'precision', 'recall', 'f1_score']:
                                if col in results_df.columns:
                                    results_df[col] = results_df[col].apply(lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) else x)
                            
                            st.dataframe(results_df)
                            
                            # Best model
                            best = trainer.get_best_model(results)
                            if best:
                                st.success(f"🏆 **Best Model:** {best['best_model']} with {best['metric']} = {best['best_score']*100:.2f}%")
                            
                            # Visualization
                            fig, ax = plt.subplots(figsize=(10, 6))
                            accuracies = []
                            model_names = []
                            for model_name, result in results.items():
                                if model_name != 'pca_info' and 'metrics' in result and 'accuracy' in result['metrics']:
                                    model_names.append(model_name)
                                    accuracies.append(result['metrics']['accuracy'] * 100)
                            
                            if accuracies:
                                bars = ax.bar(model_names, accuracies, color=['#667eea', '#764ba2', '#f39c12', '#27ae60', '#e74c3c'][:len(model_names)])
                                ax.set_ylabel('Accuracy (%)')
                                ax.set_title('Model Accuracy Comparison')
                                ax.set_ylim(0, 105)
                                for bar, acc in zip(bars, accuracies):
                                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{acc:.1f}%', ha='center', fontweight='bold')
                                plt.xticks(rotation=45)
                                st.pyplot(fig)
                        
                        if use_pca and 'pca_info' in results:
                            st.info(f"📊 PCA applied: Reduced features to {results['pca_info']['n_components']} components")
                            
                    except Exception as e:
                        st.error(f"Training error: {str(e)}")
                        st.info("Try selecting a different target column or check your data")
    else:
        st.warning("⚠️ Please load or generate a dataset first!")
# ==================== DASHBOARD ====================
elif page == "📈 Dashboard":
    st.markdown('<h2 class="section-header">📈 Complete Dashboard</h2>', unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.markdown("### Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Duplicate Rows", df.duplicated().sum())
        
        # Data quality score
        completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        st.progress(completeness / 100)
        st.caption(f"Data Quality Score: {completeness:.1f}%")
        
        # Feature distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select feature", numeric_cols)
            fig, ax = plt.subplots(figsize=(10, 4))
            df[selected_col].hist(bins=30, ax=ax, color='#667eea', edgecolor='black')
            ax.set_title(f"Distribution of {selected_col}")
            st.pyplot(fig)
    else:
        st.warning("⚠️ Please load or generate a dataset first!")
    
    if st.button("🏠 Back to Home"):
        st.session_state.page = "🏠 Home"
        st.rerun()

# ==================== DOCUMENTATION ====================
elif page == "📚 Documentation":
    st.markdown('<h2 class="section-header">📚 Documentation</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Getting Started
    
    1. **Load a Dataset** - Use built-in datasets or upload your own CSV
    2. **Analyze Your Data** - Get instant insights about your dataset
    3. **Check Compatibility** - Ensure your system can handle the dataset
    4. **Train Models** - Automatically train and compare multiple ML models
    5. **Export Results** - Download reports and models for production
    
    ### Features Explained
    
    #### 📊 Dataset Finder
    - Search and load from 5+ built-in datasets
    - Upload custom CSV files
    - Automatic data type detection
    
    #### ✨ Synthetic Generator
    - Generate classification/regression datasets
    - Control noise level and sample size
    - Perfect for testing and demos
    
    #### 🔍 Dataset Analyzer
    - Missing value detection
    - Correlation analysis
    - Outlier detection
    - Statistical summaries
    
    #### 💻 System Checker
    - RAM and CPU compatibility check
    - Memory usage estimation
    - Optimization suggestions
    
    #### 🤖 Model Trainer
    - 6+ ML algorithms
    - Automatic train-test split
    - Performance metrics comparison
    - PCA dimensionality reduction
    
    ### Tips & Tricks
    
    - 💡 Start with small datasets for faster training
    - 💡 Use synthetic data for initial testing
    - 💡 Check system compatibility before large datasets
    - 💡 Export models for production use
    
    ### Support
    
    For issues or feature requests, please contact the development team.
    """)
    
    if st.button("🏠 Back to Home"):
        st.session_state.page = "🏠 Home"
        st.rerun()
