"""
interpret.py

Functions for generating model interpretability outputs.
Uses LIME to provide word-level explanations for predictions.

Note:
We use `clean_text` (not `preprocess_and_transform`) because LIME needs
the *raw-ish* text so it can measure the effect of words.
The full preprocessing (embedding + scaling) happens inside `classifier_fn`
that LIME calls internally.
"""

import numpy as np
from lime.lime_text import LimeTextExplainer
from typing import List, Tuple
from src.preprocess import clean_text, scaler, sbert_model
from src.predict import svm_model, label_encoder

# Map long class names to shorter labels (for LIME plots)
short_map = {
    "Billing and Payments": "Bill & Pmts",
    "Customer Service, Returns & Exchanges": "CS & Returns",
    "General Inquiry": "Gen. Inq",
    "Human Resources": "HR",
    "Product Support": "Prod Support",
    "Sales and Pre-Sales": "Sales",
    "Service Outages and Maintenance": "Outages & Maint",
    "Technical & IT Support": "IT Support",
}

lime_class_names = [short_map.get(n, n) for n in label_encoder.classes_]


def explain_prediction(
    subject: str, body: str, num_features: int = 10
) -> List[Tuple[str, float]]:
    """
    Generate word-level explanations for a prediction using LIME.

    Args:
        subject (str): The subject of the support ticket.
        body (str): The body of the support ticket.
        num_features (int): Number of words to include in the explanation.

    Returns:
        List[Tuple[str, float]]: List of (word, weight) tuples showing
        which words push the prediction toward or away from the predicted class.
    """
    # Clean and combine subject/body
    new_text_cleaned = clean_text(f"{subject.strip()} {body.strip()}")

    # Classifier function for LIME
    def classifier_fn(texts):
        return svm_model.predict_proba(
            scaler.transform(
                sbert_model.encode(
                    list(texts), show_progress_bar=False, convert_to_numpy=True
                ).astype(np.float32)
            )
        )

    explainer = LimeTextExplainer(class_names=lime_class_names, random_state=42)

    # Get explanation for this instance
    exp = explainer.explain_instance(
        text_instance=new_text_cleaned,
        classifier_fn=classifier_fn,
        num_features=num_features,
        num_samples=1000,
        top_labels=1,  # only explain the top prediction
    )

    top_label = exp.available_labels()[0]
    explanation = exp.as_list(label=top_label)

    return explanation
