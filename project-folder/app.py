"""
Heart Disease Classification - Streamlit Web Application
Implements 6 ML models with interactive UI for prediction and evaluation.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Heart Disease Classification",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Classification")
st.markdown("**ML Assignment 2 — BITS Pilani M.Tech (AIML/DSE)**")
st.markdown("Comparing 6 classification models on the UCI Heart Disease Dataset.")
st.markdown("---")


# ============================================================
# HELPER FUNCTIONS
# ============================================================
@st.cache_data
def load_default_data():
    """Load the default heart.csv dataset."""
    import os
    # Get the directory where app.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_dir, "heart.csv"))
    df = df.drop_duplicates()
    return df


def get_models():
    """Return a dictionary of all 6 classification models."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100, random_state=42,
            use_label_encoder=False, eval_metric='logloss'
        )
    }


@st.cache_data
def train_all_models(df):
    """Train all models and return results, trained models, and test data."""
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = get_models()
    results = []
    all_predictions = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "AUC": round(roc_auc_score(y_test, y_prob), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1 Score": round(f1_score(y_test, y_pred), 4),
            "MCC": round(matthews_corrcoef(y_test, y_pred), 4)
        }
        results.append(metrics)
        all_predictions[name] = y_pred

    results_df = pd.DataFrame(results).set_index("Model")
    return results_df, models, all_predictions, X_test_scaled, y_test, scaler, X.columns.tolist()


# ============================================================
# SIDEBAR - DATASET UPLOAD & MODEL SELECTION
# ============================================================
st.sidebar.header("⚙️ Configuration")

# Dataset upload option
st.sidebar.subheader("📂 Upload Test Data (CSV)")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file (must have same features as heart.csv)",
    type=["csv"]
)

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Uploaded: {uploaded_file.name} ({uploaded_df.shape[0]} rows)")
else:
    uploaded_df = None

# Model selection dropdown
st.sidebar.subheader("🤖 Select Model")
model_names = [
    "All Models",
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
]
selected_model = st.sidebar.selectbox("Choose a model to view details:", model_names)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** UCI Heart Disease")
st.sidebar.markdown("**Instances:** 1025 | **Features:** 13")
st.sidebar.markdown("**Task:** Binary Classification")


# ============================================================
# MAIN SECTION - LOAD DATA & TRAIN MODELS
# ============================================================
df = load_default_data()

# Train models
with st.spinner("Training all 6 models..."):
    results_df, models, all_predictions, X_test_scaled, y_test, scaler, feature_names = train_all_models(df)

# ============================================================
# TAB LAYOUT
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset Overview",
    "📈 Model Comparison",
    "🔍 Model Details",
    "📂 Uploaded Data Predictions"
])

# ============================================================
# TAB 1: DATASET OVERVIEW
# ============================================================
with tab1:
    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Instances", df.shape[0])
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("Heart Disease (1)", int(df['target'].sum()))
    col4.metric("No Disease (0)", int((df['target'] == 0).sum()))

    st.subheader("First 10 Rows")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Target Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='target', data=df, palette='Set2', ax=ax)
    ax.set_xticklabels(["No Disease (0)", "Disease (1)"])
    ax.set_title("Target Class Distribution")
    st.pyplot(fig)

    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix")
    st.pyplot(fig)

# ============================================================
# TAB 2: MODEL COMPARISON
# ============================================================
with tab2:
    st.header("📈 Model Comparison - All 6 Models")

    # Metrics comparison table
    st.subheader("Evaluation Metrics Comparison Table")
    st.dataframe(
        results_df.style.highlight_max(axis=0, color='lightgreen')
            .highlight_min(axis=0, color='#ffcccc')
            .format("{:.4f}"),
        use_container_width=True
    )

    # Best model highlight
    best_model = results_df['F1 Score'].idxmax()
    best_f1 = results_df.loc[best_model, 'F1 Score']
    st.success(f"🏆 Best model by F1 Score: **{best_model}** ({best_f1:.4f})")

    # Bar chart comparison
    st.subheader("Performance Comparison Chart")
    fig, ax = plt.subplots(figsize=(14, 6))
    results_df.plot(kind='bar', ax=ax, colormap='Set2', edgecolor='black')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    # Confusion matrices for all models
    st.subheader("Confusion Matrices - All Models")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (name, model) in enumerate(models.items()):
        y_pred = all_predictions[name]
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=["No Disease", "Disease"],
                    yticklabels=["No Disease", "Disease"])
        axes[idx].set_title(f'{name}', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================
# TAB 3: INDIVIDUAL MODEL DETAILS
# ============================================================
with tab3:
    st.header("🔍 Individual Model Details")

    if selected_model == "All Models":
        st.info("👈 Select a specific model from the sidebar to view its details.")
    else:
        st.subheader(f"Model: {selected_model}")

        # Metrics for selected model
        model_metrics = results_df.loc[selected_model]

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{model_metrics['Accuracy']:.4f}")
        col2.metric("AUC Score", f"{model_metrics['AUC']:.4f}")
        col3.metric("Precision", f"{model_metrics['Precision']:.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{model_metrics['Recall']:.4f}")
        col5.metric("F1 Score", f"{model_metrics['F1 Score']:.4f}")
        col6.metric("MCC", f"{model_metrics['MCC']:.4f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        y_pred = all_predictions[selected_model]
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=["No Disease", "Disease"],
                    yticklabels=["No Disease", "Disease"])
        ax.set_title(f'Confusion Matrix - {selected_model}', fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

        # Classification Report
        st.subheader("Classification Report")
        report = classification_report(
            y_test, y_pred,
            target_names=["No Disease", "Disease"],
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

# ============================================================
# TAB 4: UPLOADED DATA PREDICTIONS
# ============================================================
with tab4:
    st.header("📂 Predictions on Uploaded Data")

    if uploaded_df is not None:
        st.subheader("Uploaded Data Preview")
        st.dataframe(uploaded_df.head(10), use_container_width=True)

        # Check if target column exists
        has_target = 'target' in uploaded_df.columns

        if has_target:
            X_uploaded = uploaded_df.drop('target', axis=1)
            y_uploaded = uploaded_df['target']
        else:
            X_uploaded = uploaded_df
            y_uploaded = None

        # Validate columns
        missing_cols = set(feature_names) - set(X_uploaded.columns)
        if missing_cols:
            st.error(f"❌ Missing columns in uploaded data: {missing_cols}")
        else:
            X_uploaded = X_uploaded[feature_names]
            X_uploaded_scaled = scaler.transform(X_uploaded)

            # Model selection for prediction
            pred_model_name = st.selectbox(
                "Select model for prediction:",
                list(models.keys()),
                key="pred_model"
            )

            model = models[pred_model_name]
            y_pred_uploaded = model.predict(X_uploaded_scaled)
            y_prob_uploaded = model.predict_proba(X_uploaded_scaled)[:, 1]

            # Show predictions
            result_df = X_uploaded.copy()
            result_df['Prediction'] = y_pred_uploaded
            result_df['Prediction Label'] = result_df['Prediction'].map(
                {0: "No Disease", 1: "Disease"}
            )
            result_df['Probability'] = np.round(y_prob_uploaded, 4)

            st.subheader(f"Predictions using {pred_model_name}")
            st.dataframe(result_df, use_container_width=True)

            # Summary
            col1, col2 = st.columns(2)
            col1.metric("Predicted Disease", int((y_pred_uploaded == 1).sum()))
            col2.metric("Predicted No Disease", int((y_pred_uploaded == 0).sum()))

            # If target exists, show evaluation metrics
            if has_target and y_uploaded is not None:
                st.subheader(f"Evaluation Metrics on Uploaded Data")

                acc = accuracy_score(y_uploaded, y_pred_uploaded)
                auc = roc_auc_score(y_uploaded, y_prob_uploaded)
                prec = precision_score(y_uploaded, y_pred_uploaded)
                rec = recall_score(y_uploaded, y_pred_uploaded)
                f1 = f1_score(y_uploaded, y_pred_uploaded)
                mcc = matthews_corrcoef(y_uploaded, y_pred_uploaded)

                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{acc:.4f}")
                col2.metric("AUC", f"{auc:.4f}")
                col3.metric("Precision", f"{prec:.4f}")

                col4, col5, col6 = st.columns(3)
                col4.metric("Recall", f"{rec:.4f}")
                col5.metric("F1 Score", f"{f1:.4f}")
                col6.metric("MCC", f"{mcc:.4f}")

                # Confusion matrix for uploaded data
                st.subheader("Confusion Matrix - Uploaded Data")
                cm = confusion_matrix(y_uploaded, y_pred_uploaded)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=["No Disease", "Disease"],
                            yticklabels=["No Disease", "Disease"])
                ax.set_title(f'Confusion Matrix - {pred_model_name} (Uploaded Data)', fontweight='bold')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)

                # Classification Report
                st.subheader("Classification Report - Uploaded Data")
                report = classification_report(
                    y_uploaded, y_pred_uploaded,
                    target_names=["No Disease", "Disease"],
                    output_dict=True
                )
                st.dataframe(
                    pd.DataFrame(report).transpose().style.format("{:.4f}"),
                    use_container_width=True
                )
    else:
        st.info("👈 Upload a CSV file from the sidebar to see predictions.")
        st.markdown("""
        **Expected CSV format:**
        - Must contain the same 13 feature columns as the heart disease dataset
        - Optionally include a `target` column for evaluation
        - Features: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`
        """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "Built with ❤️ using Streamlit | "
    "ML Assignment 2 - BITS Pilani M.Tech (AIML/DSE)"
)