import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score

st.title("Income Classification App")

st.write("Upload test dataset and choose model for prediction.")


# Load models
models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Decision Tree": joblib.load("models/decision_tree_classifier.pkl"),
    "KNN": joblib.load("models/k-nearest_neighbor_classifier.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes_classifier_-_gaussian_or_multinomial.pkl"),
    "Random Forest": joblib.load("models/ensemble_model_-_random_forest.pkl"),
    "XGBoost": joblib.load("models/ensemble_model_-_xgboost.pkl"),
}

# Model selection
model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]


# Upload dataset
uploaded_file = st.file_uploader("Upload CSV test dataset(Income)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview")
    st.write(df.head())

    # Cleaning
    df = df.replace("?", np.nan)
    df = df.dropna()

    if "income" not in df.columns:
        st.error("Dataset must contain 'income' column.")
    elif st.button("Perform Evaluation"):
        try:
            # Target encoding
            y = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)
            X = df.drop("income", axis=1)

             # Encoding
            categorical_cols = X.select_dtypes(include="object").columns
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

            # Align columns (important)
            model_features = model.feature_names_in_
            X = X.reindex(columns=model_features, fill_value=0)

            # Scaling
            X = scaler.transform(X)

            # Predictions
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]

            # Metrics
            acc = accuracy_score(y, y_pred)
            prec = precision_score(y, y_pred)
            rec = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            mcc = matthews_corrcoef(y, y_pred)
            auc = roc_auc_score(y, y_prob)

            st.subheader(f"Model Performance for {model_name}")

            st.write(f"Accuracy: {acc:.4f}")
            st.write(f"AUC: {auc:.4f}")
            st.write(f"Precision: {prec:.4f}")
            st.write(f"Recall: {rec:.4f}")
            st.write(f"F1 Score: {f1:.4f}")
            st.write(f"MCC: {mcc:.4f}")

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y, y_pred)
            st.write(cm)
        except Exception as e:
            st.error(f"Error processing given file: {e}")
else:
    st.info("Please upload the test data in csv format")

st.sidebar.markdown("📧 [Email Me](mailto:2025aa05902@wilp.bits-pilani.ac.in)")

