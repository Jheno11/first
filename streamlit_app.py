import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt

# App Title
st.title("🎈 My Streamlit App for Cancer Classification")

# File Upload
uploaded_file = st.file_uploader("Choose the datasetWhiteMerged.csv file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded CSV File:")
    st.dataframe(df)

    # Data Processing
    data = df.round()

    # Create Conditions
    conditions = ['cancer' if '-01' in sample else 'normal' for sample in data.columns]
    data.loc['Condition'] = conditions

    # Display Condition Counts
    condition_counts = data.loc['Condition'].value_counts()
    st.write("Number of normal samples:", condition_counts['normal'])
    st.write("Number of cancer samples:", condition_counts['cancer'])
    st.write("Total number of samples:", int(condition_counts.sum()))

    # Transpose Data
    data = data.T
    data = data.iloc[1:, :]

    # Encode Target Variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['Condition'])

    # Prepare Features
    X = np.asarray(data.iloc[:, :-1])

    # Resample Dataset
    def scale_dataset(X, y, SMOTEEN=False):
        if SMOTEEN:
            sme = SMOTEENN(sampling_strategy=0.32, random_state=42)
            X, y = sme.fit_resample(X, y)
        return X, y

    X, y = scale_dataset(X, y, SMOTEEN=True)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training with Hyperparameter Tuning
    hyperparameters = {
        'var_smoothing': np.logspace(0, -9, num=100)
    }
    estimator = GaussianNB()
    grid_search = GridSearchCV(estimator, hyperparameters, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model_gaussian_nb = grid_search.best_estimator_

    st.write(f"Best parameters found: {grid_search.best_params_}")
    st.write(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    # Predictions and Metrics
    nb_y_pred = best_model_gaussian_nb.predict(X_test)
    nb_y_pred_train = best_model_gaussian_nb.predict(X_train)

    st.write("Test Set Classification Report")
    st.text(classification_report(y_test, nb_y_pred))

    st.write("Train Set Classification Report")
    st.text(classification_report(y_train, nb_y_pred_train))

    # Confusion Matrix
    st.write("Confusion Matrix for Test Set")
    cm = confusion_matrix(y_test, nb_y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, values_format='d')
    plt.title("Confusion Matrix - Naive Bayes")
    st.pyplot(fig)
