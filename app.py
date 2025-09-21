import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.predict import predict_ticket, label_encoder
from src.interpret import explain_prediction

# Streamlit app config
st.set_page_config(page_title="Support Ticket Queue Predictor", layout="centered")

st.title("Support Ticket Queue Predictor")
st.write("Enter a support ticket subject and body to predict the most likely queue.")

# --- Input fields ---
subject = st.text_input("Ticket Subject")
body = st.text_area("Ticket Body", height=150)

if st.button("Predict"):
    if subject.strip() == "" and body.strip() == "":
        st.warning("Please enter a subject or body text.")
    else:
        with st.spinner("Analyzing ticket..."):
            # Run prediction
            predictions = predict_ticket(subject, body)
            explanation = explain_prediction(subject, body, num_features=10)

        # --- Results Section ---
        st.subheader("Predicted Queues")

        # Progress bar based on top prediction confidence
        top_score = predictions[0][1]
        st.progress(int(top_score * 100))

        # Show top 3 predictions as selectable buttons
        st.write("### Suggested Queues")
        cols = st.columns(len(predictions))
        selected_queue = None
        for i, (queue, score) in enumerate(predictions):
            if cols[i].button(f"{queue} ({score:.2f})"):
                selected_queue = queue

        # Dropdown — default to None
        st.write("### Final Queue Assignment")
        queue_options = ["-- Select a queue --"] + list(label_encoder.classes_)
        assigned_queue = st.selectbox(
            "Choose the queue",
            options=queue_options,
            index=0,  # Default to placeholder
        )

        # Confirmation message when queue selected
        if assigned_queue != "-- Select a queue --":
            st.success(f"Queue assignment confirmed: **{assigned_queue}**")

        elif selected_queue:  # if button was pressed but dropdown unchanged
            st.info(f"Suggested queue selected: **{selected_queue}**")

        # --- Explanation Section ---
        st.subheader("Interpretability (Top Words)")

        if explanation:
            # Convert explanation to DataFrame for plotting
            df = pd.DataFrame(explanation, columns=["word", "weight"])
            df = df.sort_values("weight", ascending=True)  # for horizontal bar plot

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(
                df["word"],
                df["weight"],
                color=["#1f77b4" if w > 0 else "#d72e2e" for w in df["weight"]],
            )
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Weight")
            ax.set_title("Word Contributions to Prediction")
            st.pyplot(fig)

            # Also print raw explanation values
            st.write("**Top Words & Weights**")
            for word, weight in explanation:
                st.write(f"- {word}: {weight:.4f}")
