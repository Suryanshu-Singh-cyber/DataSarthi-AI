import streamlit as st
import pandas as pd

# Import modules
from modules.data_generator import generate_data
from modules.data_analyzer import analyze_data, plot_correlation, data_quality_score
from modules.model_trainer import train_models, apply_pca
from modules.system_checker import check_system
from modules.recommender import recommend_model

# Page config
st.set_page_config(page_title="DataSarthi AI", layout="wide")

# Title
st.title("🧠 DataSarthi AI")
st.markdown("### Intelligent Dataset Assistant 🚀")

# Sidebar
menu = st.sidebar.selectbox(
    "Select Feature",
    ["Home", "Generate Data", "Analyze Data", "Train Models", "System Checker", "Model Recommendation"]
)

st.sidebar.info("Built by Suryanshu Singh 🚀")
st.sidebar.success("ML Project - 4th Semester")

# ---------------- HOME ----------------
if menu == "Home":
    st.write("Welcome to DataSarthi AI 🎯")
    st.write("""
    This tool helps you:
    - Generate synthetic datasets
    - Analyze datasets
    - Train ML models
    - Check system compatibility
    - Recommend best ML models
    """)

# ---------------- GENERATE DATA ----------------
elif menu == "Generate Data":
    st.header("🧬 Synthetic Data Generator")

    rows = st.slider("Number of Rows", 100, 10000, 500)
    cols = st.slider("Number of Features", 2, 20, 5)

    if st.button("Generate Data"):
        df = generate_data(rows, cols)
        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "synthetic_data.csv")

# ---------------- ANALYZE DATA ----------------
elif menu == "Analyze Data":
    st.header("📊 Dataset Analyzer")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.subheader("Preview")
        st.dataframe(df.head())

        # Metrics
        score = data_quality_score(df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Quality", score)

        # Analysis
        report = analyze_data(df)

        st.subheader("Basic Info")
        st.write("Shape:", report['shape'])
        st.write("Missing Values:", report['missing'])

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        fig = plot_correlation(df)
        st.pyplot(fig)

# ---------------- TRAIN MODELS ----------------
elif menu == "Train Models":
    st.header("🤖 Model Training & Comparison")

    uploaded_file = st.file_uploader("Upload CSV for Training", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("Dataset Loaded ✅")

        if st.checkbox("Apply PCA"):
            X = df.iloc[:, :-1]
            X_pca = apply_pca(X)
            st.write("PCA Output (first 5 rows):")
            st.write(X_pca[:5])

        if st.button("Train Models"):
            results = train_models(df)

            for model, res in results.items():
                st.subheader(model)
                st.write("Accuracy:", round(res["accuracy"], 3))
                st.write("Detailed Report:")
                st.write(res["report"])

# ---------------- SYSTEM CHECKER ----------------
elif menu == "System Checker":
    st.header("⚙️ System Compatibility Checker")

    rows = st.number_input("Dataset Rows", 100, 1000000, 1000)
    cols = st.number_input("Dataset Columns", 1, 100, 10)
    ram = st.slider("RAM (GB)", 2, 32, 8)

    if st.button("Check Compatibility"):
        result = check_system(rows, cols, ram)
        st.write(result)

# ---------------- MODEL RECOMMENDER ----------------
elif menu == "Model Recommendation":
    st.header("🧠 Model Recommendation Engine")

    rows = st.number_input("Dataset Rows", 100, 1000000, 1000)
    cols = st.number_input("Number of Features", 1, 100, 10)

    if st.button("Recommend Model"):
        model = recommend_model(rows, cols)
        st.success(f"Recommended Model: {model}")
