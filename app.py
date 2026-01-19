import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="ML Assignment-2", layout="wide")
st.title("📊 ML Assignment-2: Classification Dashboard")

st.caption("Upload TEST data only. Target column is optional.")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

model_map = {
    "Logistic Regression": "logistic",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost"
}

model_name = st.selectbox("Select Model", list(model_map.keys()))

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    target_col = "default.payment.next.month"

    if target_col in data.columns:
        X = data.drop(target_col, axis=1)
        y = data[target_col]
        show_metrics = True
    else:
        X = data
        show_metrics = False

    model = joblib.load(f"models/saved_models/{model_map[model_name]}.pkl")
    y_pred = model.predict(X)

    st.subheader("🔍 Sample Predictions")
    st.write(y_pred[:10])

    if show_metrics:
        st.subheader("📌 Classification Report")
        st.text(classification_report(y, y_pred))

        st.subheader("📌 Confusion Matrix")
        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
    else:
        st.info("ℹ️ Target column not found. Showing predictions only.")
