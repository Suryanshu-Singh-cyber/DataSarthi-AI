import streamlit as st

st.title("🧠 DataSarthi AI")
st.write("Your Intelligent Dataset Assistant 🚀")

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
