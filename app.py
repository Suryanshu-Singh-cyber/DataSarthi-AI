import streamlit as st

st.set_page_config(page_title="DataSarthi AI", layout="wide")

st.title("🧠 DataSarthi AI")
st.markdown("### Your Intelligent Dataset Assistant 🚀")

menu = st.sidebar.selectbox(
    "Select Feature",
    ["Home", "Generate Data", "Analyze Data", "Train Models", "System Checker", "Model Recommendation"]
)
if menu == "Home":
    st.write("Welcome to DataSarthi AI")

st.sidebar.info("Built by Suryanshu Singh 🚀")
st.sidebar.success("4th Semester ML Project")

from modules.data_generator import generate_data

st.header("🧬 Synthetic Data Generator")

rows = st.slider("Number of Rows", 100, 10000, 500)
cols = st.slider("Number of Features", 2, 20, 5)

if st.button("Generate Data"):
    df = generate_data(rows, cols)
    st.write(df.head())

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "synthetic_data.csv")

from modules.data_analyzer import analyze_data

st.header("📊 Dataset Analyzer")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    report = analyze_data(df)

    st.write("Shape:", report['shape'])
    st.write("Missing Values:", report['missing'])

from modules.model_trainer import train_models

if uploaded_file:
    if st.button("Train Models"):
        results = train_models(df)
        st.write(results)

from modules.system_checker import check_system

st.header("⚙️ System Checker")

ram = st.slider("RAM (GB)", 2, 32, 8)

if st.button("Check Compatibility"):
    result = check_system(rows, cols, ram)
    st.write(result)

from modules.recommender import recommend_model

if st.button("Recommend Model"):
    model = recommend_model(rows, cols)
    st.write("Recommended Model:", model)
