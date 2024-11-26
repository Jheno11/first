import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# App Title
st.title("🎈 My Streamlit App for Cancer Classification")

# File Upload
uploaded_file = st.file_uploader("Choose a CSV file for processing", type="csv")

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded CSV File:")
        st.dataframe(df)

        # Check if the data has the expected structure
        if len(df.columns) < 2:  # Minimum expected columns
            st.error("The uploaded file must have at least two columns.")
        else:
            # Process Data
            data = df.round()

            # Create Conditions
            try:
                conditions = ['cancer' if '-01' in sample else 'normal' for sample in data.columns]
                data.loc['Condition'] = conditions
            except Exception as e:
                st.error("Error in condition assignment: Ensure the column names include '-01' for 'cancer'.")
                st.stop()

            # Display Condition Counts
            condition_counts = data.loc['Condition'].value_counts()
            st.write("Number of normal samples:", condition_counts.get('normal', 0))
            st.write("Number of cancer samples:", condition_counts.get('cancer', 0))
            st.write("Total number of samples:", int(condition_counts.sum()))

            # Transpose Data
            data = data.T
            data = data.iloc[1:, :]  # Exclude the 'Condition' row for features

            # Encode Target Variable
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(data['Condition'])

            # Prepare Features
            X = np.asarray(data.iloc[:, :-1])  # Exclude the 'Condition' column for features

            # Display Data Information
            st.write("Features (X):")
            st.dataframe(X)
            st.write("Labels (y):")
            st.dataframe(y)

            # Show Train-Test Split Information
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            st.write("Training set:")
            st.write("In y_train: Cancers =", sum(y_train == 0), ", Normals =", sum(y_train == 1))

            st.write("\nTest set:")
            st.write("In y_test: Cancers =", sum(y_test == 0), ", Normals =", sum(y_test == 1))

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
