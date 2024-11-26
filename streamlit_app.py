import streamlit as st
import pandas as pd

st.title("🎈 My new app is streamlit")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

st.title("Upload and Display datasetWhiteMerged.csv")

uploaded_file = st.file_uploader("Choose the datasetWhiteMerged.csv file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded CSV File:")
    st.dataframe(df)
