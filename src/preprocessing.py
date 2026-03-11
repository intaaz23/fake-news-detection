"""Text Preprocessing Module for Fake News Detection.

This module provides functions to clean and preprocess text data
for NLP-based fake news classification.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Clean and preprocess a single text string.

    Steps:
        1. Convert to lowercase
        2. Remove URLs
        3. Remove special characters and numbers
        4. Remove extra whitespace
        5. Remove stopwords
        6. Apply lemmatization

    Args:
        text (str): Raw text string to clean.

    Returns:
        str: Cleaned and preprocessed text.
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def preprocess_dataframe(df, text_column='text', label_column='label'):
    """Preprocess an entire DataFrame for model training.

    Args:
        df (pd.DataFrame): DataFrame containing text and label columns.
        text_column (str): Name of the text column. Default 'text'.
        label_column (str): Name of the label column. Default 'label'.

    Returns:
        pd.DataFrame: DataFrame with cleaned text and validated labels.
    """
    import pandas as pd

    print(f"[INFO] Original dataset shape: {df.shape}")

    # Drop rows with missing text or label
    df = df.dropna(subset=[text_column, label_column])
    print(f"[INFO] After dropping NaN: {df.shape}")

    # Remove duplicate rows
    df = df.drop_duplicates(subset=[text_column])
    print(f"[INFO] After removing duplicates: {df.shape}")

    # Clean the text column
    print("[INFO] Cleaning text... (this may take a few minutes)")
    df['cleaned_text'] = df[text_column].apply(clean_text)

    # Remove empty cleaned text
    df = df[df['cleaned_text'].str.strip().astype(bool)]
    print(f"[INFO] After cleaning: {df.shape}")

    return df