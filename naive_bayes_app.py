import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="GaussianNB Classifier", layout="wide")

st.title("🧠 Gaussian Naive Bayes Classifier")

# --- SIDEBAR: INPUTS ---
with st.sidebar:
    st.header("1. Data & Split")
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    
    test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
    random_state = st.number_input("Random Seed", value=42)

# --- MAIN PAGE: LOGIC ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    # Select features and target
    columns = df.columns.tolist()
    target_col = st.selectbox("Select Target Column (Y)", columns, index=len(columns)-1)
    feature_cols = st.multiselect("Select Feature Columns (X)", columns, default=[c for c in columns if c != target_col])

    if st.button("Train Model") and feature_cols:
        X = df[feature_cols]
        y = df[target_col]

        X = pd.get_dummies(X, drop_first=True)
        
        # 1. Training/Testing Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 2. Model Initialization & Training
        model = GaussianNB()
        model.fit(X_train, y_train)

        # 3. Predictions
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        # --- OUTPUTS ---
        st.divider()
        col1, col2 = st.columns(2)
        
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)

        col1.metric("Training Accuracy", f"{train_acc:.2%}")
        col2.metric("Testing Accuracy", f"{test_acc:.2%}")

        # 4. Confusion Matrix Visualization
        st.write("### Confusion Matrix (Testing Data)")
        cm = confusion_matrix(y_test, test_preds)
        
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        st.pyplot(fig)

else:
    st.info("Please upload a CSV file in the sidebar to get started!")

