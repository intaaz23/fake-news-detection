"""Feature Extraction Module for Fake News Detection.

This module provides functions to convert preprocessed text data
into numerical features using TF-IDF and Count Vectorization.
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
import os

def extract_tfidf_features(train_text, test_text, max_features=5000, ngram_range=(1, 2)):
    """Extract TF-IDF features from text data.

    Args:
        train_text (pd.Series): Training text data.
        test_text (pd.Series): Testing text data.
        max_features (int): Maximum number of features. Default 5000.
        ngram_range (tuple): Range of n-grams. Default (1, 2).

    Returns:
        tuple: (X_train_tfidf, X_test_tfidf, tfidf_vectorizer)
    """
    print(f"[INFO] Extracting TF-IDF features (max_features={max_features}, ngram_range={ngram_range})")

    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        max_df=0.95,
        min_df=2
    )

    X_train = tfidf_vectorizer.fit_transform(train_text)
    X_test = tfidf_vectorizer.transform(test_text)

    print(f"[INFO] TF-IDF Train shape: {X_train.shape}")
    print(f"[INFO] TF-IDF Test shape: {X_test.shape}")

    return X_train, X_test, tfidf_vectorizer

def extract_count_features(train_text, test_text, max_features=5000, ngram_range=(1, 2)):
    """Extract Count Vectorizer features from text data.

    Args:
        train_text (pd.Series): Training text data.
        test_text (pd.Series): Testing text data.
        max_features (int): Maximum number of features. Default 5000.
        ngram_range (tuple): Range of n-grams. Default (1, 2).

    Returns:
        tuple: (X_train_count, X_test_count, count_vectorizer)
    """
    print(f"[INFO] Extracting Count features (max_features={max_features}, ngram_range={ngram_range})")

    count_vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english'
    )

    X_train = count_vectorizer.fit_transform(train_text)
    X_test = count_vectorizer.transform(test_text)

    print(f"[INFO] Count Train shape: {X_train.shape}")
    print(f"[INFO] Count Test shape: {X_test.shape}")

    return X_train, X_test, count_vectorizer

def save_vectorizer(vectorizer, filepath='models/tfidf_vectorizer.pkl'):
    """Save a fitted vectorizer to disk.

    Args:
        vectorizer: Fitted vectorizer object.
        filepath (str): Path to save the vectorizer.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"[INFO] Vectorizer saved to {filepath}")

def load_vectorizer(filepath='models/tfidf_vectorizer.pkl'):
    """Load a saved vectorizer from disk.

    Args:
        filepath (str): Path to the saved vectorizer.

    Returns:
        Fitted vectorizer object.
    """
    with open(filepath, 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"[INFO] Vectorizer loaded from {filepath}")
    return vectorizer