#!/usr/bin/env python3
"""
Train a logistic regression classifier on embeddings.
"""

import json
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_data():
    """Load training and test data."""
    X_train = np.load("train_embeddings_X.npy")
    y_train = np.load("train_embeddings_y.npy")
    X_test = np.load("test_embeddings_X.npy")
    y_test = np.load("test_embeddings_y.npy")

    print(f"Training data: {X_train.shape[0]} examples, {X_train.shape[1]} dimensions")
    print(f"Test data: {X_test.shape[0]} examples")
    print(f"Training class distribution: {np.sum(y_train == 1)} questions, {np.sum(y_train == 0)} keywords")

    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train):
    """Train logistic regression classifier."""
    print("\nTraining logistic regression classifier...")

    model = LogisticRegression(
        max_iter=1000,
        C=1.0,  # Regularization strength
        class_weight='balanced',  # Handle any class imbalance
        random_state=42
    )

    model.fit(X_train, y_train)

    # Training accuracy
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"Training accuracy: {train_acc:.4f}")

    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data."""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.1f}%)")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["keyword", "question"]))

    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                Predicted")
    print(f"              keyword  question")
    print(f"Actual keyword    {cm[0][0]:4d}     {cm[0][1]:4d}")
    print(f"Actual question   {cm[1][0]:4d}     {cm[1][1]:4d}")

    return acc

def analyze_model(model):
    """Analyze model weights."""
    print("\n" + "="*50)
    print("MODEL ANALYSIS")
    print("="*50)

    weights = model.coef_[0]
    bias = model.intercept_[0]

    print(f"\nWeight vector shape: {weights.shape}")
    print(f"Bias: {bias:.6f}")
    print(f"Weight stats: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")

    # Most important dimensions
    top_positive = np.argsort(weights)[-10:][::-1]
    top_negative = np.argsort(weights)[:10]

    print(f"\nTop 10 dimensions for 'question' classification:")
    for idx in top_positive:
        print(f"  dim {idx}: {weights[idx]:.4f}")

    print(f"\nTop 10 dimensions for 'keyword' classification:")
    for idx in top_negative:
        print(f"  dim {idx}: {weights[idx]:.4f}")

def save_model(model):
    """Save model in multiple formats."""
    # Pickle format (for Python use)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("\nSaved model.pkl")

    # JSON format (for Rust integration)
    weights = model.coef_[0].tolist()
    bias = float(model.intercept_[0])

    model_json = {
        "weights": weights,
        "bias": bias,
        "threshold": 0.5,
        "labels": ["keyword", "question"],
        "embedding_dim": len(weights),
        "description": "Query intent classifier: question vs keyword search"
    }

    with open("model_weights.json", "w") as f:
        json.dump(model_json, f, indent=2)
    print("Saved model_weights.json")

    # Print size
    import os
    size_bytes = os.path.getsize("model_weights.json")
    print(f"Model size: {size_bytes / 1024:.1f} KB")

def test_examples(model, X_test, y_test):
    """Test on some specific examples and show results."""
    with open("test_embeddings.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)

    # Get predictions with probabilities
    y_proba = model.predict_proba(X_test)

    # Show some examples
    indices = np.random.choice(len(test_data), size=min(20, len(test_data)), replace=False)

    for idx in indices:
        ex = test_data[idx]
        prob_question = y_proba[idx][1]
        predicted = "question" if prob_question > 0.5 else "keyword"
        actual = ex["label"]
        status = "✓" if predicted == actual else "✗"

        print(f"{status} [{prob_question:.2f}] {predicted:8s} | {ex['query'][:60]}")

def main():
    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate
    accuracy = evaluate_model(model, X_test, y_test)

    # Analyze
    analyze_model(model)

    # Save
    save_model(model)

    # Test examples
    test_examples(model, X_test, y_test)

    print("\n" + "="*50)
    print(f"TRAINING COMPLETE - Accuracy: {accuracy*100:.1f}%")
    print("="*50)
    print("\nNext step: Copy model_weights.json to your Rust project")

if __name__ == "__main__":
    main()
