"""Prediction Module for Fake News Detection.

This module provides functions to predict whether a given
news article is real or fake using a trained model.
"""

import pickle
from src.preprocessing import clean_text

def load_pipeline(model_path='models/best_model.pkl', vectorizer_path='models/tfidf_vectorizer.pkl'):
    """Load the trained model and vectorizer.

    Args:
        model_path (str): Path to the saved model.
        vectorizer_path (str): Path to the saved vectorizer.

    Returns:
        tuple: (model, vectorizer)
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def predict_news(text, model, vectorizer):
    """Predict whether a news article is real or fake.

    Args:
        text (str): Raw news article text.
        model: Trained ML model.
        vectorizer: Fitted TF-IDF vectorizer.

    Returns:
        dict: Prediction result with label and confidence.
    """
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]

    confidence = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)[0]
        confidence = max(probabilities) * 100

    label = 'REAL' if prediction == 1 else 'FAKE'

    result = {
        'prediction': label,
        'prediction_code': int(prediction),
        'confidence': f"{confidence:.2f}%" if confidence else "N/A",
        'cleaned_text': cleaned,
    }

    return result

def predict_batch(texts, model, vectorizer):
    """Predict labels for multiple news articles.

    Args:
        texts (list): List of raw news article strings.
        model: Trained ML model.
        vectorizer: Fitted TF-IDF vectorizer.

    Returns:
        list: List of prediction result dicts.
    """
    results = []
    for text in texts:
        result = predict_news(text, model, vectorizer)
        results.append(result)
    return results

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("   FAKE NEWS DETECTOR - Quick Test")
    print("=" * 50)

    model, vectorizer = load_pipeline()

    sample_text = input("\nEnter a news headline or article: ")
    result = predict_news(sample_text, model, vectorizer)

    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")
