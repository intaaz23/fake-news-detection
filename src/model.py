"""Model Training and Evaluation Module for Fake News Detection.

This module provides functions to train multiple ML models,
evaluate their performance, and save the best model.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

def get_models():
    """Return a dictionary of ML models to train and compare.

    Returns:
        dict: Dictionary of model name to model instance.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Linear SVC': LinearSVC(max_iter=2000, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Passive Aggressive': PassiveAggressiveClassifier(max_iter=1000, random_state=42),
    }
    return models

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train multiple models and evaluate their performance.

    Args:
        X_train: Training features (sparse matrix).
        X_test: Testing features (sparse matrix).
        y_train: Training labels.
        y_test: Testing labels.

    Returns:
        tuple: (results_dict, best_model_tuple)
    """
    models = get_models()
    results = {}
    best_accuracy = 0
    best_model = None

    print("\n" + "=" * 60)
    print("       MODEL TRAINING & EVALUATION RESULTS")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n[TRAINING] {name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, target_names=['FAKE', 'REAL'])
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'report': report,
            'confusion_matrix': cm,
            'model': model,
        }

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = (name, model)

    print("\n" + "=" * 60)
    print(f"  BEST MODEL: {best_model[0]} (Accuracy: {best_accuracy:.4f})")
    print("=" * 60)

    return results, best_model

def save_model(model, filepath='models/best_model.pkl'):
    """Save a trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"[INFO] Model saved to {filepath}")

def load_model(filepath='models/best_model.pkl'):
    """Load a saved model from disk."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"[INFO] Model loaded from {filepath}")
    return model

def plot_accuracy_comparison(results, save_path='reports/accuracy_comparison.png'):
    """Plot a bar chart comparing model accuracies."""
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] * 100 for name in names]

    plt.figure(figsize=(12, 6))
    bars = plt.barh(names, accuracies, color=sns.color_palette('viridis', len(names)))
    plt.xlabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison - Fake News Detection')
    plt.xlim(0, 100)

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{acc:.2f}%', ha='left', va='center', fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"[INFO] Accuracy comparison chart saved to {save_path}")

def plot_confusion_matrix(cm, model_name, save_path='reports/'):
    """Plot a confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(filepath, dpi=150)
    plt.show()
    print(f"[INFO] Confusion matrix saved to {filepath}")

