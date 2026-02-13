"""
Heart Disease Classification - Model Training Script
Dataset: UCI Heart Disease Dataset (1025 instances, 13 features)
Models: Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Evaluation
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)


def load_and_preprocess_data(filepath):
    """Load dataset, remove duplicates, and return features and target."""
    df = pd.read_csv(filepath)

    print(f"Dataset Shape: {df.shape}")
    print(f"Number of Features: {df.shape[1] - 1}")
    print(f"Number of Instances: {df.shape[0]}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nStatistical Summary:\n{df.describe()}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nTarget Distribution:\n{df['target'].value_counts()}")
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")

    # Remove duplicates
    df = df.drop_duplicates()
    print(f"Shape after removing duplicates: {df.shape}")

    return df


def split_and_scale(df):
    """Split data into train/test and apply Standard Scaling."""
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


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


def evaluate_model(model, X_test, y_test):
    """Calculate all 6 evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1 Score": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4)
    }, y_pred


def train_and_evaluate_all(X_train, X_test, y_train, y_test):
    """Train all 6 models and return results and trained models."""
    models = get_models()
    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\n--- {name} ---")

        # Train
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Evaluate
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        metrics["Model"] = name
        results.append(metrics)

        # Print metrics
        for k, v in metrics.items():
            if k != "Model":
                print(f"  {k}: {v}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"  Confusion Matrix:\n{cm}")

        # Classification Report
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    return results, trained_models


def plot_confusion_matrices(trained_models, X_test, y_test):
    """Plot confusion matrices for all models."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (name, model) in enumerate(trained_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=["No Disease", "Disease"],
                    yticklabels=["No Disease", "Disease"])
        axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Confusion matrices saved to confusion_matrices.png")


def plot_model_comparison(results_df):
    """Plot bar chart comparing all models across metrics."""
    fig, ax = plt.subplots(figsize=(14, 6))
    results_df.plot(kind='bar', ax=ax, colormap='Set2')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Model comparison chart saved to model_comparison.png")


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":

    # 1. Load and explore data
    print("=" * 60)
    print("1. LOADING AND EXPLORING DATASET")
    print("=" * 60)
    # Read from parent folder (project root) where heart.csv is stored
    df = load_and_preprocess_data(os.path.join(os.path.dirname(__file__), "..", "heart.csv"))

    # 2. Preprocess
    print("\n" + "=" * 60)
    print("2. DATA PREPROCESSING")
    print("=" * 60)
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)

    # 3. Train and Evaluate all models
    print("\n" + "=" * 60)
    print("3. MODEL TRAINING AND EVALUATION")
    print("=" * 60)
    results, trained_models = train_and_evaluate_all(X_train, X_test, y_train, y_test)

    # 4. Comparison Table
    print("\n" + "=" * 60)
    print("4. MODEL COMPARISON TABLE")
    print("=" * 60)
    results_df = pd.DataFrame(results).set_index("Model")
    print(f"\n{results_df.to_string()}")

    # 5. Visualizations
    print("\n" + "=" * 60)
    print("5. GENERATING VISUALIZATIONS")
    print("=" * 60)
    plot_confusion_matrices(trained_models, X_test, y_test)
    plot_model_comparison(results_df)

    # 6. Save test data for Streamlit upload
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    X_test_df['target'] = y_test.values
    X_test_df.to_csv("test_data.csv", index=False)
    print(f"\nTest data saved to test_data.csv ({X_test_df.shape[0]} rows)")

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)