#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# MLflow integration
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from mlflow.tracking import MlflowClient
import os

# save model in pickle format
import pickle
import datetime

# Load data
data = pd.read_csv("./data/heart.csv")

# Handle missing values
data = data.fillna(data.median(numeric_only=True))

# Identify features and target
X = data.drop("target", axis=1)
y = data["target"]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Simple encoding and scaling
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ]
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Heart_Disease_Prediction")

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_auc = 0
best_run = None
best_pipe = None

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        pipe = Pipeline([
            ('preprocess', preprocessor),
            ('clf', model)
        ])
        # Log model parameters
        mlflow.log_params(model.get_params())
        # Cross-validation
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc')
        cv_acc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy')
        cv_prec = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='precision')
        cv_rec = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='recall')
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        
        test_metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_roc_auc": roc_auc_score(y_test, y_prob)
        }
        
        results[name] = {
            "cv_auc_mean": np.mean(cv_auc),
            "cv_accuracy_mean": np.mean(cv_acc),
            "cv_precision_mean": np.mean(cv_prec),
            "cv_recall_mean": np.mean(cv_rec),
            **test_metrics
        }

        # Log metrics
        mlflow.log_metric("cv_auc_mean", np.mean(cv_auc))
        mlflow.log_metric("cv_accuracy_mean", np.mean(cv_acc))
        mlflow.log_metric("cv_precision_mean", np.mean(cv_prec))
        mlflow.log_metric("cv_recall_mean", np.mean(cv_rec))
        for k, v in test_metrics.items():
            mlflow.log_metric(k, v)

        # Ensure the output directory exists
        output_dir = "./output/"
        os.makedirs(output_dir, exist_ok=True)

        # Log confusion matrix plot
        plt.figure()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.title(f"Confusion Matrix - {name}")
        cm_plot_path = os.path.join(output_dir, f"confusion_matrix_{name.replace(' ', '_')}.png")
        plt.savefig(cm_plot_path)
        mlflow.log_artifact(cm_plot_path)
        plt.close()

        # Log ROC curve plot
        plt.figure()
        RocCurveDisplay.from_predictions(y_test, y_prob)
        plt.title(f"ROC Curve - {name}")
        roc_plot_path = os.path.join(output_dir, f"roc_curve_{name.replace(' ', '_')}.png")
        plt.savefig(roc_plot_path)
        mlflow.log_artifact(roc_plot_path)
        plt.close()

        # Log model
        mlflow.sklearn.log_model(pipe, "model")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"model_{name.replace(' ', '_')}_{timestamp}"
        mlflow.sklearn.save_model(pipe, path=model_path)

        if test_metrics["test_roc_auc"] > best_auc:
            best_auc = test_metrics["test_roc_auc"]
            best_run = mlflow.active_run().info.run_id
            best_pipe = pipe

# Document results
for model_name, metrics in results.items():
    print(f"\nModel: {model_name}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

# Find the best model based on test ROC AUC
best_model = max(results.items(), key=lambda x: x[1]['test_roc_auc'])
print(f"\nBest Model: {best_model[0]} with Test ROC AUC: {best_model[1]['test_roc_auc']:.4f}")

# Register best model in MLflow Model Registry with automated versioning
model_registry_name = "HeartDiseaseModel"
if best_run is not None:
    client = MlflowClient()
    # Check if model already exists in registry
    try:
        latest_versions = client.get_latest_versions(model_registry_name)
        if latest_versions:
            # Get the highest version number
            latest_version = max(int(v.version) for v in latest_versions)
            next_version = latest_version + 1
        else:
            next_version = 1
    except Exception:
        # Model does not exist yet
        next_version = 1

    # Register model
    result = mlflow.register_model(
        model_uri=f"runs:/{best_run}/model",
        name=model_registry_name
    )

    # Save the final best model in pickle format with version in filename
    model_dir = "./model"
    os.makedirs(model_dir, exist_ok=True)
    with open(f"{model_dir}/best_heart_disease_model_v{next_version}.pkl", "wb") as f:
        pickle.dump(best_pipe, f)
else:
    # Save as default if not registered
    with open("./model/best_heart_disease_model.pkl", "wb") as f:
        pickle.dump(best_pipe, f)
