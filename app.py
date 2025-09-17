# app.py
import streamlit as st
import joblib

st.set_page_config(page_title="SMS Spam Classifier")
st.title("SMS Spam Classifier Demo")

# Load trained model
model = joblib.load('models/spam_classifier.joblib') # Ensure the model is in the 'models' directory

# Text input for user
msg = st.text_area("Enter your SMS message here:", height=150)

# Prediction button
if st.button("Predict"): 
    if not msg.strip():
        st.warning("Please enter a message.")
    else:
        pred = model.predict([msg])[0]
        proba = model.predict_proba([msg])[0][1]  # probability of spam
        label = "SPAM" if pred == 1 else "NOT SPAM"
        st.subheader(f"Prediction: {label}")
        st.write(f"Spam probability: {proba:.2f}")
