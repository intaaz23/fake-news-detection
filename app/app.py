"""Streamlit Web Application for Fake News Detection.

Usage:
    streamlit run app/app.py
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import clean_text
from src.predict import load_pipeline, predict_news

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .result-fake {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
        margin: 10px 0;
    }
    .result-real {
        background-color: #ccffcc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00cc00;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>📰 Fake News Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>MCA Major Project 2026 | ML & NLP Based</p>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app uses **Machine Learning** and **NLP** to detect **Real** or **Fake** news.
    
    **Tech Stack:** Python, Scikit-learn, NLTK, Streamlit
    
    **Models:** Logistic Regression, Naive Bayes, Random Forest, Linear SVC, Gradient Boosting, Passive Aggressive
    """
    )
    st.markdown("---")
    st.header("📊 How It Works")
    st.write("""
    1. Paste a news article
    2. Text is cleaned & normalized
    3. TF-IDF vectorization
    4. ML model classifies the text
    5. Real or Fake with confidence
    """
    )

@st.cache_resource
def load_model_and_vectorizer():
    try:
        model, vectorizer = load_pipeline(
            model_path='models/best_model.pkl',
            vectorizer_path='models/tfidf_vectorizer.pkl'
        )
        return model, vectorizer, True
    except FileNotFoundError:
        return None, None, False

model, vectorizer, model_loaded = load_model_and_vectorizer()

if not model_loaded:
    st.error("""
    ⚠️ **Model not found!**
    
    Please train the model first by running:
    ```bash
    python main.py
    ```
    """
    )
else:
    st.success("✅ Model loaded successfully! Ready to detect fake news.")
    
    st.subheader("📝 Enter News Article")
    news_input = st.text_area(
        "Paste a news article or headline below:",
        height=200,
        placeholder="Enter the news article text here..."
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔍 Detect Fake News", use_container_width=True, type="primary")

    if predict_button:
        if news_input.strip():
            with st.spinner("Analyzing the article..."):
                result = predict_news(news_input, model, vectorizer)

            st.markdown("---")
            st.subheader("📊 Prediction Result")

            if result['prediction'] == 'FAKE':
                st.markdown("""
                <div class='result-fake'>
                    <h2>🚨 FAKE NEWS DETECTED</h2>
                    <p>This article appears to be <strong>FAKE</strong>.</p>
                </div>
                """, unsafe_allow_html=True)
                st.error(f"📊 Confidence: {result['confidence']}")
            else:
                st.markdown("""
                <div class='result-real'>
                    <h2>✅ REAL NEWS</h2>
                    <p>This article appears to be <strong>REAL</strong>.</p>
                </div>
                """, unsafe_allow_html=True)
                st.success(f"📊 Confidence: {result['confidence']}")

            with st.expander("🔍 View Processed Text"):
                st.write(result['cleaned_text'])
        else:
            st.warning("⚠️ Please enter some text to analyze.")

    st.markdown("---")
    st.subheader("💡 Try Sample News")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🟢 Try Real News Sample", use_container_width=True):
            sample_real = "The government announced new economic reforms aimed at boosting GDP growth and creating more employment opportunities across the country."
            result = predict_news(sample_real, model, vectorizer)
            st.info(f"**Sample:** {sample_real[:100]}...")
            st.success(f"**Prediction:** {result['prediction']} | **Confidence:** {result['confidence']}")
    
    with col2:
        if st.button("🔴 Try Fake News Sample", use_container_width=True):
            sample_fake = "BREAKING: Scientists have confirmed that eating chocolate every day can make you immortal according to a secret NASA study!"
            result = predict_news(sample_fake, model, vectorizer)
            st.info(f"**Sample:** {sample_fake[:100]}...")
            st.error(f"**Prediction:** {result['prediction']} | **Confidence:** {result['confidence']}")

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;'>"
    "Built with ❤️ by intaaz23 | MCA Major Project 2026 | "
    "Python, Scikit-learn, NLTK & Streamlit"
    "</p>",
    unsafe_allow_html=True
)