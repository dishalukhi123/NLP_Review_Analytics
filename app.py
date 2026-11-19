"""Streamlit app: Movie Review Sentiment Analyzer.

Run from the folder that contains the `models` folder (D:\\NLP):
    streamlit run app.py
Requires models/sentiment_pipeline.pkl, created in Step 4 of the notebook.
"""
import math
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models") / "sentiment_pipeline.pkl"

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="🎬", layout="centered")


# ---------- preprocessing (identical to the notebook's clean_text) ----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", " emailtoken ", text)
    text = re.sub(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b", " phonetoken ", text)
    text = re.sub(r"\b\d+\b", " numbertoken ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------- cached resources ----------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        return None


@st.cache_resource
def load_pos_tagger():
    """Return a function (text -> list of (word, tag)) or None if NLTK data is missing."""
    try:
        import nltk
        for res in ("punkt", "punkt_tab", "averaged_perceptron_tagger_eng"):
            nltk.download(res, quiet=True)
        from nltk import pos_tag, word_tokenize
        pos_tag(word_tokenize("test"))  # fails fast if resources are unavailable
        return lambda t: pos_tag(word_tokenize(t))
    except Exception:
        return None


def predict(model, text):
    """Return (label, confidence 0-1) for the given raw review text."""
    cleaned = clean_text(text)
    label = model.predict([cleaned])[0]
    if hasattr(model, "predict_proba"):
        confidence = float(max(model.predict_proba([cleaned])[0]))
    elif hasattr(model, "decision_function"):
        # LinearSVC has no probabilities; map the margin to 0-1 with a sigmoid.
        margin = abs(float(model.decision_function([cleaned])[0]))
        confidence = 1 / (1 + math.exp(-margin))
    else:
        confidence = float("nan")
    return label, confidence


def vader_label(analyzer, text):
    compound = analyzer.polarity_scores(text)["compound"]
    if compound >= 0.05:
        return "positive", compound
    if compound <= -0.05:
        return "negative", compound
    return "neutral", compound


# ---------- UI ----------
st.title("🎬 Movie Review Sentiment Analyzer")
st.caption("A TF-IDF + machine-learning model trained on 50,000 IMDb reviews.")

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: `{MODEL_PATH}`. Run the notebook through Step 4 "
        "(the cell that saves the model), then start this app from the same folder."
    )
    st.stop()

model = load_model()

EXAMPLES = {
    "Positive example": "The movie was fantastic. The story and acting were excellent.",
    "Negative example": "This film was boring and a complete waste of time.",
    "Neutral example": "The movie was released last week.",
}

with st.sidebar:
    st.header("Options")
    show_vader = st.checkbox("Compare with VADER", value=True)
    show_pos = st.checkbox("Show POS tags", value=False)
    show_clean = st.checkbox("Show cleaned text", value=False)
    st.divider()
    st.subheader("Try an example")
    for name, text in EXAMPLES.items():
        if st.button(name, use_container_width=True):
            st.session_state["review_text"] = text

review = st.text_area("Enter a movie review", key="review_text", height=160,
                      placeholder="Type or paste a review here...")

if st.button("Analyze", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        label, confidence = predict(model, review)
        icon = "😊" if label == "positive" else "😞"
        st.subheader(f"{icon} Predicted sentiment: {label.capitalize()}")
        if not math.isnan(confidence):
            st.progress(min(max(confidence, 0.0), 1.0), text=f"Model confidence: {confidence:.0%}")
            if confidence < 0.65:
                st.info("Low confidence. The text may be neutral or mixed. "
                        "This model only knows positive and negative.")

        if show_clean:
            st.markdown("**Cleaned text**")
            st.code(clean_text(review), language=None)

        if show_vader:
            analyzer = load_vader()
            if analyzer is None:
                st.warning("VADER is not installed: `pip install vaderSentiment`.")
            else:
                v_label, v_score = vader_label(analyzer, review)
                st.markdown("**Method comparison**")
                st.table(pd.DataFrame({
                    "Method": ["ML model (TF-IDF + classifier)", "VADER (rule/lexicon based)"],
                    "Prediction": [label, v_label],
                    "Detail": [f"confidence {confidence:.0%}", f"compound {v_score:+.2f}"],
                }))

        if show_pos:
            tagger = load_pos_tagger()
            if tagger is None:
                st.warning("POS tagging needs NLTK data (punkt and the perceptron tagger) "
                           "and could not be loaded.")
            else:
                st.markdown("**Part-of-speech tags**")
                st.dataframe(pd.DataFrame(tagger(review), columns=["word", "POS_tag"]),
                             use_container_width=True, hide_index=True)

st.divider()
st.caption("College NLP practical project. Binary sentiment only (positive/negative).")
