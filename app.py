import streamlit as st
import streamlit.components.v1 as components
from src.predict import predict_ticket, label_encoder
from src.interpret import explain_prediction

# Streamlit app config
st.set_page_config(page_title="Support Ticket Queue Predictor", layout="centered")

st.title("Support-Ticket Automated Routing Assistant (SARA)")
st.write("Enter a support ticket subject and body to predict the most likely queue.")

# --- Input fields ---
subject = st.text_input("Ticket Subject")
body = st.text_area("Ticket Body", height=150)

if st.button("Predict"):
    if subject.strip() == "" and body.strip() == "":
        st.warning("Please enter a subject or body text.")
    else:
        with st.spinner("Analyzing ticket..."):
            # Run prediction + interpretability
            predictions = predict_ticket(subject, body)
            explanation, exp = explain_prediction(subject, body, num_features=10)

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

        # Dropdown — default to placeholder
        st.write("### Final Queue Assignment")
        queue_options = ["-- Select a queue --"] + list(label_encoder.classes_)
        assigned_queue = st.selectbox(
            "Choose the queue",
            options=queue_options,
            index=0,  # Default to placeholder
        )

        # --- Explanation Section ---
        st.subheader("Explaination")

        st.markdown(
            """
            The chart below explains why the model chose a specific queue.  

            - **Prediction probabilities (top-left bar chart):** shows how confident the model 
            is in each possible queue. Higher values mean more confidence.  

            - **Word importance chart (top-right plot):** shows which words pushed the 
            decision *toward* or *away from* the predicted queue. Words on the right 
            support the prediction; words on the left suggest another queue.  

            - **Highlighted text (bottom view):** shows the original ticket with the 
            most influential words highlighted. Darker highlights mean stronger influence.  
            """
        )

        # Wrap LIME HTML in white background
        lime_html = exp.as_html(text=True, predict_proba=True)
        wrapped_html = f"""
        <div style="background-color:white; color:black;">
            {lime_html}
        </div>
        """
        components.html(wrapped_html, height=1500)
