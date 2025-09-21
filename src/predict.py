import numpy as np
from typing import List, Tuple
from src.preprocess import preprocess_and_transform
import joblib


# load saved model and components
svm_model = joblib.load("models/svm.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")


def predict_ticket(subject: str, body: str) -> List[Tuple[str, float]]:
    """
    Predict the top 3 most likely support queues for a ticket using the trained model.

    This function passes the subject and body through the preprocessing pipeline
    (combine, redact, clean, embed, scale) and then uses the trained classifier
    to return the top 3 predicted queue labels along with their confidence scores.

    Args:
        subject (str): The subject line of the support ticket.
        body (str): The body text of the support ticket.

    Returns:
        List[Tuple[str, float]]: A list of up to 3 tuples containing:
            - The predicted queue label (str).
            - The model confidence score for that label (float between 0 and 1).
            Sorted in descending order of confidence.
    """

    # Preprocess and transform into scaled embedding
    transformed = preprocess_and_transform(subject, body)

    # Predict probabilities
    proba = svm_model.predict_proba(transformed)[0]

    # Get indices of top 3 probabilities
    top_indices = np.argsort(proba)[::-1][:3]

    # Decode labels and pair with scores
    top_predictions = [
        (label_encoder.inverse_transform([idx])[0], float(proba[idx]))
        for idx in top_indices
    ]

    return top_predictions
