import os
import pickle
import pandas as pd
import streamlit as st
from utils.helpers import setup_logging
from transformers import pipeline

logger = setup_logging()

CRIME_CATEGORIES = ['theft', 'assault', 'accident', 'drug_crime', 'cybercrime', 'non_crime']
SVM_MODEL_PATH = "models/baseline_svm.pkl"
_muril_classifier = None

@st.cache_resource(show_spinner=False)
def get_kannada_classifier():
    """Load the pre-trained MuRIL classifier for Kannada text."""
    global _muril_classifier
    import warnings
    # Suppress transformers warnings in production
    warnings.filterwarnings("ignore")
    
    if _muril_classifier is None:
        try:
            logger.info("Loading fine-tuned MuRIL classifier from models/saved_muril...")
            from models.muril_classifier import MuRILClassifier
            # Instantiate using local saved model and weights
            _muril_classifier = MuRILClassifier(CRIME_CATEGORIES, model_name="models/saved_muril")
        except Exception as e:
            logger.error(f"Failed to load MuRIL model: {e}")
            
    return _muril_classifier

def load_svm_model():
    """Load the saved TF-IDF + SVM pipeline from disk."""
    if not os.path.exists(SVM_MODEL_PATH):
        logger.warning(f"SVM model not found at {SVM_MODEL_PATH}. Run main.py --use-synthetic first.")
        return None, None
    try:
        with open(SVM_MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        logger.info("SVM model loaded for real-time inference.")
        return bundle["vectorizer"], bundle["classifier"]
    except Exception as e:
        logger.error(f"Failed to load SVM model: {e}")
        return None, None


def classify_articles(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """
    Classify articles using SVM for English and Zero-Shot Transformer for Kannada.
    Adds multi-hot crime category columns to the DataFrame.
    """
    vectorizer, classifier = load_svm_model()

    # Initialise all category columns to 0 as default
    for cat in CRIME_CATEGORIES:
        df[cat] = 0

    if df.empty:
        return df

    # Helper function to detect Kannada text naively (by unicode range check)
    def is_kannada(text):
        return any('\u0C80' <= char <= '\u0CFF' for char in str(text))

    try:
        muril = get_kannada_classifier()
        
        for idx, row in df.iterrows():
            text = str(row[text_col]).strip()
            if not text:
                continue
                
            if is_kannada(text):
                # Run Kannada text through the fine-tuned MuRIL transformer
                try:
                    if muril:
                        # predict returns a list of assigned labels like [['accident']]
                        muril_labels = muril.predict([text])[0]
                        for label in muril_labels:
                            if label in CRIME_CATEGORIES:
                                df.at[idx, label] = 1
                        logger.debug(f"MuRIL mapped snippet to -> {muril_labels}")
                except Exception as e:
                    logger.error(f"MuRIL classification error on snippet: {e}")
            else:
                # Run English text through fast SVM
                if vectorizer and classifier:
                    X = vectorizer.transform([text])
                    preds = classifier.predict(X)[0]
                    for i, cat in enumerate(CRIME_CATEGORIES):
                        df.at[idx, cat] = preds[i]

        logger.info(f"Classified {len(df)} articles using Hybrid AI Pipeline (SVM + MuRIL).")
    except Exception as e:
        logger.error(f"Classification pipeline failed: {e}")

    return df
