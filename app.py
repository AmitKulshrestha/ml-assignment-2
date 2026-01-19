import streamlit as st
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="ML Assignment-2",
    layout="wide"
)

st.title("📊 ML Assignment-2: Classification Dashboard")
st.caption("Upload TEST data only. Target column is optional.")

# --------------------------------------------------
# Upload section
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

# --------------------------------------------------
# Model selection
# --------------------------------------------------
model_map = {
    "Logistic Regression": "logistic",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost"
}

model_name = st.selectbox(
    "Select Machine Learning Model",
    list(model_map.keys())
)

# --------------------------------------------------
# Main logic
# --------------------------------------------------
if uploaded_file is not None:

    # Load uploaded data
    data = pd.read_csv(uploaded_file)

    target_col = "default.payment.next.month"

    # Check if target exists
    if target_col in data.columns:
        X = data.drop(columns=[target_col])
        y = data[target_col]
        has_target = True
    else:
        X = data
        y = None
        has_target = False

    # Model path
    model_path = f"models/saved_models/{model_map[model_name]}.pkl"

    # Safety check for model file
    if not os.path.exists(model_path):
        st.error("❌ Trained model file not found.")
        st.write("Expected path:", model_path)
        st.write("👉 Make sure `.pkl` files are committed to GitHub.")
        st.stop()

    # Load model
    model = joblib.load(model_path)

    # Predict
    y_pred = model.predict(X)

    # --------------------------------------------------
    # Output
    # --------------------------------------------------
    st.subheader("🔍 Sample Predictions")
    st.write(pd.DataFrame({
        "Prediction": y_pred[:10]
    }))

    # --------------------------------------------------
    # Evaluation (only if target exists)
    # --------------------------------------------------
    if has_target:
        st.subheader("📌 Classification Report")
        st.text(classification_report(y, y_pred))

        st.subheader("📌 Confusion Matrix")
        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)
    else:
        st.info(
            "ℹ️ Target column not found in uploaded data. "
            "Showing predictions only (as required by assignment)."
        )
