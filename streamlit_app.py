import streamlit as st
import pandas as pd

st.title("🎈 Prostate Cancer")
st.write(
    "a demo for preprocessing and modeling of prostate cancer dataset"
)

st.title("Upload and Display dataset")

uploaded_file = st.file_uploader("Choose the dataset (csv)", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded CSV File:")
    st.dataframe(df)


