import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

# Load the pre-trained model
model = load_model('final_model.keras')

tokenizer = joblib.load('tokenizer.pkl')

# Function to preprocess text
def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
    return padded_sequence

# Streamlit app UI
st.title("Text Sentiment Prediction")
st.write("Enter some text to classify its sentiment (positive/negative)")

# Input text
input_text = st.text_area("Your text:")

# Predict button
if st.button('Predict'):
    if input_text:
        # Preprocess the input text
        preprocessed_text = preprocess_text(input_text)
        
        # Make a prediction
        prediction = model.predict(preprocessed_text)
                
        # Output prediction result
        if prediction[0] > 0.5:
            st.error("This text is **negative**.")
        else:
            st.success("This text is **positive**.")
    else:
        st.warning("Please enter some text to predict.")
