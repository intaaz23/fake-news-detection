"""Main Pipeline Runner for Fake News Detection Project.

This script runs the complete ML pipeline:
    1. Load dataset
    2. Preprocess text
    3. Extract features (TF-IDF)
    4. Train & evaluate multiple models
    5. Save the best model & vectorizer
    6. Generate reports & charts

Usage:
    python main.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import preprocess_dataframe
from src.feature_extraction import extract_tfidf_features, save_vectorizer
from src.model import train_and_evaluate, save_model, plot_accuracy_comparison, plot_confusion_matrix
from src.predict import predict_news

def main():
    """Run the complete Fake News Detection pipeline."""

    print("\n" + "=" * 60)
    print("   FAKE NEWS DETECTION - MCA Major Project")
    print("   Complete ML Pipeline")
    print("=" * 60)

    # STEP 1: Load Dataset
    print("\n[STEP 1/6] Loading dataset...")
    
    data_path = 'data/train.csv'
    
    if not os.path.exists(data_path):
        print(f"\n[ERROR] Dataset not found at '{data_path}'")
        print("Please download the dataset from:")
        print("  https://www.kaggle.com/c/fake-news/data")
        print("\nPlace 'train.csv' inside the 'data/' folder.")
        return
    
    df = pd.read_csv(data_path)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[INFO] Columns: {list(df.columns)}")
    print(f"[INFO] Label distribution:\n{df['label'].value_counts()}")

    # STEP 2: Preprocess Text
    print("\n[STEP 2/6] Preprocessing text data...")

    if 'title' in df.columns and 'text' in df.columns:
        df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        text_column = 'combined_text'
    elif 'text' in df.columns:
        text_column = 'text'
    else:
        print("[ERROR] No 'text' column found in dataset.")
        return

    df = preprocess_dataframe(df, text_column=text_column, label_column='label')

    # STEP 3: Train/Test Split
    print("\n[STEP 3/6] Splitting data into train/test sets...")

    X = df['cleaned_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[INFO] Training set: {X_train.shape[0]} samples")
    print(f"[INFO] Testing set:  {X_test.shape[0]} samples")

    # STEP 4: Feature Extraction (TF-IDF)
    print("\n[STEP 4/6] Extracting TF-IDF features...")

    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = extract_tfidf_features(
        X_train, X_test, max_features=5000, ngram_range=(1, 2)
    )

    save_vectorizer(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')

    # STEP 5: Train & Evaluate Models
    print("\n[STEP 5/6] Training and evaluating models...")

    results, best_model = train_and_evaluate(
        X_train_tfidf, X_test_tfidf, y_train, y_test
    )

    save_model(best_model[1], 'models/best_model.pkl')

    # STEP 6: Generate Reports & Charts
    print("\n[STEP 6/6] Generating reports and charts...")

    plot_accuracy_comparison(results, save_path='reports/accuracy_comparison.png')

    best_name = best_model[0]
    best_cm = results[best_name]['confusion_matrix']
    plot_confusion_matrix(best_cm, best_name, save_path='reports/')

    print(f"\n{'=' * 60}")
    print(f"  DETAILED REPORT - {best_name}")
    print(f"{'=' * 60}")
    print(results[best_name]['report'])

    # QUICK TEST
    print("\n" + "=" * 60)
    print("   QUICK PREDICTION TEST")
    print("=" * 60)

    sample_fake = "BREAKING: Scientists discover that the moon is made of cheese!"
    sample_real = "The government announced new economic policies to boost GDP growth."

    result_fake = predict_news(sample_fake, best_model[1], tfidf_vectorizer)
    result_real = predict_news(sample_real, best_model[1], tfidf_vectorizer)

    print(f"\n  Sample 1: '{sample_fake[:60]}...'\n  Prediction: {result_fake['prediction']} | Confidence: {result_fake['confidence']}")

    print(f"\n  Sample 2: '{sample_real[:60]}...'\n  Prediction: {result_real['prediction']} | Confidence: {result_real['confidence']}")

    print("\n" + "=" * 60)
    print("   PIPELINE COMPLETE! All models trained and saved.")
    print("   Run 'streamlit run app/app.py' to launch the web app.")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()