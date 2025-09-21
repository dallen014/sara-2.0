import numpy as np
from src.preprocess import preprocess_and_transform
import joblib


# load saved model and components
svm_model = joblib.load("../models/svm.joblib")
label_encoder = joblib.load("../models/label_encoder.joblib")


def predict_ticket(subject: str, body: str) -> tuple[str, float]:
    """
    Predict the most likely support queue for a ticket using the trained model.

    This function passes the subject and body through the preprocessing pipeline
    (combine, redact, clean, embed, scale) and then uses the trained classifier
    to return the predicted queue label along with its confidence score.

    Args:
        subject (str): The subject line of the support ticket.
        body (str): The body text of the support ticket.

    Returns:
        tuple[str, float]: A tuple containing:
            - The predicted queue label (str).
            - The model confidence score for that label (float between 0 and 1).
    """
    # Preprocess and transform into scaled embedding
    transformed = preprocess_and_transform(subject, body)

    # Predict probabilities
    proba = svm_model.predict_proba(transformed)[0]

    # Get index of highest probability
    pred_idx = np.argmax(proba)

    # Decode predicted label
    queue = label_encoder.inverse_transform([pred_idx])[0]
    score = proba[pred_idx]

    return queue, score
