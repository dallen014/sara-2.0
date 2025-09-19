"""
preprocess.py

Pipeline for preparing support ticket text before model predictions.
Steps:
1. Combine subject and body
2. Redact PII
3. Clean text
4. Embed with SBERT
5. Scale
"""

import re
import joblib
import spacy
import numpy as np
from typing import Any
from sentence_transformers import SentenceTransformer

# Load English NLP model for PII redaction
nlp_en = spacy.load("en_core_web_lg")

# Load saved scaler
scaler = joblib.load("models/svm_scaler.joblib")

# Load SBERT model
sbert_model = SentenceTransformer("all-mpnet-base-v2")


def combine_subject_body(subject: str, body: str) -> str:
    """
    Combine subject and body into a single string.

    Args:
        subject (str): Ticket subject.
        body (str): Ticket body.

    Returns:
        str: Combined subject + body text.
    """
    return f"{subject.strip()}. {body.strip()}"


def redact_pii(text: Any) -> str:
    """
    Redact PII from text using regex and Named Entity Recognition (NER).

    Regex:
      - Emails
      - Phone numbers
      - IP addresses
      - Credit card numbers
      - Street-style addresses

    NER:
      - PERSON (names)

    Args:
        text (Any): Input text to redact.
        lang (str): Language code ('en' or 'de') for appropriate NER model.

    Returns:
        str: Text with PII replaced by placeholders.
    """
    if not isinstance(text, str):
        return ""

    redacted = text

    # regex patterns for personal identifiable information (PII)
    patterns = {
        "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "phone": (
            r"\b(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?|\d{3})" r"[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ),
        "ip": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
        "address": r"\b\d{1,5}\s+\w+(?:\s\w+)?\s+(St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane)\b",
    }
    # apply regex patterns to redact PII
    for key, pattern in patterns.items():
        redacted = re.sub(pattern, f"[{key.upper()}_REDACTED]", redacted)

    # NER-based redaction
    doc = nlp_en(redacted)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            redacted = redacted.replace(ent.text, "<NAME>")

    return redacted


def clean_text(text):
    """
    Clean text minimally for transformer-based embeddings.

    Performs minimal cleaning appropriate for transformer models:
      - Strips leading and trailing whitespace.
      - Normalizes internal whitespace to a single space.
      - Removes line breaks.
      - Optionally removes HTML tags.

    Args:
        text (Any):The input text to be cleaned.

    Returns
        str: Cleaned text string .
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()

    # replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)

    # remove line breaks
    text = text.replace("\n", " ").replace("\r", " ")
    return text


def embed_text(text: str) -> np.ndarray:
    """
    Embed text using the SBERT model.

    Args:
        text (str): Cleaned input text.

    Returns:
        np.ndarray: Embedding vector.
    """
    return sbert_model.encode([text])[0]  # shape (768,) for mpnet


def preprocess_and_transform(subject: str, body: str) -> np.ndarray:
    """
    Full preprocessing pipeline for inference:
    1. Combine subject + body
    2. Redact PII
    3. Clean text
    4. Embed
    5. Scale

    Args:
        subject (str): Ticket subject text.
        body (str): Ticket body text.

    Returns:
        np.ndarray: Final scaled embedding ready for model prediction.
    """

    # Step 1: Combine
    combined = combine_subject_body(subject, body)

    # Step 2: Redact PII
    redacted = redact_pii(combined)

    # Step 3: Clean text
    cleaned = clean_text(redacted)

    # Step 4: Embed
    embedding = embed_text(cleaned)

    # Step 5: Scale
    transformed = scaler.transform([embedding])

    return transformed
