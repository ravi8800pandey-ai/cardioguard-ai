"""
Cardiovascular Disease Prediction - Model Training & Export Script
Extracts and serializes the best-performing model (Random Forest Classifier)
along with the StandardScaler preprocessing pipeline and evaluation metrics.
"""

import os
import json
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.pipeline import Pipeline
import joblib

SEED = 42

def load_and_preprocess_data(data_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load raw dataset, clean physiologically impossible values, and engineer features."""
    print(f"Loading raw data from: {data_path}")
    df = pd.read_csv(data_path, sep=';')
    print(f"Raw shape: {df.shape}")

    # Convert age (days -> years) and drop id
    df['age_years'] = (df['age'] / 365.0).round(1)
    df.drop(columns=['id', 'age'], inplace=True)

    # Remove physiologically impossible values matching notebook criteria
    df = df[
        (df['ap_hi'] >= 60)  & (df['ap_hi'] <= 250) &
        (df['ap_lo'] >= 40)  & (df['ap_lo'] <= 160) &
        (df['height']>= 100) & (df['height']<= 250) &
        (df['weight']>= 30)  & (df['weight']<= 200)
    ]

    # Feature Engineering
    df['bmi'] = (df['weight'] / ((df['height'] / 100.0) ** 2)).round(2)
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

    print(f"Cleaned shape: {df.shape}")

    X = df.drop(columns=['cardio'])
    y = df['cardio']
    return X, y

def train_and_export(data_path: str, output_dir: str):
    """Train the best model pipeline, evaluate, and save artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "best_rf_pipeline.joblib")
    metrics_path = os.path.join(output_dir, "model_metrics.json")

    X, y = load_and_preprocess_data(data_path)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # Build scikit-learn Pipeline with StandardScaler and the best Random Forest parameters
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=SEED,
            n_jobs=-1
        ))
    ])

    print("Training Random Forest Pipeline...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    training_time = round(time.time() - t0, 2)
    print(f"Training completed in {training_time}s")

    # Predict on test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    rf_estimator = pipeline.named_steps['rf']
    feat_importances = {
        feat: round(float(imp), 4)
        for feat, imp in zip(feature_names, rf_estimator.feature_importances_)
    }
    sorted_importances = dict(sorted(feat_importances.items(), key=lambda item: item[1], reverse=True))

    metrics = {
        "model_name": "Random Forest Classifier",
        "parameters": {
            "n_estimators": 100,
            "max_depth": 8,
            "random_state": SEED
        },
        "training_time_seconds": training_time,
        "accuracy": round(float(acc) * 100, 2),
        "roc_auc": round(float(roc_auc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm,
        "feature_names": feature_names,
        "feature_importances": sorted_importances,
        "test_size": len(y_test),
        "train_size": len(y_train),
        "benchmark_comparison": [
            {"model": "Random Forest", "accuracy": 74.09, "status": "Best Model 🏆"},
            {"model": "Decision Tree", "accuracy": 73.77, "status": "Runner Up"},
            {"model": "Logistic Regression", "accuracy": 73.37, "status": "Baseline"},
            {"model": "SVM (LinearSVC)", "accuracy": 73.24, "status": "Baseline"},
            {"model": "KNN (k=11)", "accuracy": 72.40, "status": "Baseline"}
        ]
    }

    # Save pipeline and metrics
    print(f"Saving pipeline to: {model_path}")
    joblib.dump(pipeline, model_path)

    print(f"Saving metrics to: {metrics_path}")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)

    print("=" * 60)
    print("MODEL TRAINING & SERIALIZATION SUMMARY")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']}%")
    print(f"ROC-AUC:   {metrics['roc_auc']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall:    {metrics['recall']}")
    print(f"F1-Score:  {metrics['f1_score']}")
    print("\nTop 5 Important Features:")
    for k, v in list(sorted_importances.items())[:5]:
        print(f"  {k:<20} {v}")
    print("=" * 60)
    print("Artifacts successfully generated!")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Look for dataset in project root or current folder
    dataset_candidates = [
        os.path.join(current_dir, "..", "cardio_train.csv"),
        os.path.join(current_dir, "cardio_train.csv"),
        r"d:\RAVI DOC\projects\cardio_train.csv"
    ]
    data_file = None
    for path in dataset_candidates:
        if os.path.exists(path):
            data_file = os.path.abspath(path)
            break

    if not data_file:
        raise FileNotFoundError("Could not locate cardio_train.csv dataset.")

    output_dir = os.path.join(current_dir, "models")
    train_and_export(data_file, output_dir)
