import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# App Title
st.title("🎈 Prostate Cancer")
st.write("A demo for preprocessing and modeling of prostate cancer dataset")

# File Upload
st.title("Upload and Display dataset")

uploaded_file = st.file_uploader("Choose the dataset (csv)", type="csv")

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded CSV File:")
        st.dataframe(df)

        # Preprocessing steps
        data = df.round()  # Round the data to avoid float precision issues

        # Create 'Condition' row (assuming '-01' indicates cancer and others are normal)
        conditions = ['cancer' if '-01' in sample else 'normal' for sample in data.columns]
        data.loc['Condition'] = conditions

        # Display condition counts
        condition_counts = data.loc['Condition'].value_counts()
        st.write(f"Number of normal samples: {condition_counts.get('normal', 0)}")
        st.write(f"Number of cancer samples: {condition_counts.get('cancer', 0)}")
        st.write(f"Total number of samples: {int(condition_counts.sum())}")

        # Transpose the data
        data = data.T
        st.write("Transposed Data:")
        st.dataframe(data)

        # Remove the first row which is 'Condition' for features
        data = data.iloc[1:, :]  # Exclude the 'Condition' row for features
        st.write("Data without the 'Condition' row:")
        st.dataframe(data)

        # Encode the target variable (Condition)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(data['Condition'])

        # Display the encoded labels (y)
        st.write("Encoded Labels (y):")
        st.write(y)

        # Prepare features (X)
        X = np.asarray(data.iloc[:, :-1])  # Exclude the 'Condition' column for features

        # Display the features (X)
        st.write("Features (X):")
        st.dataframe(X)

        # Training set information
        st.write("Training set:")
        st.write(f"In y: Cancers = {sum(y == 0)}, Normals = {sum(y == 1)}")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
