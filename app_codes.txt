import json
import pickle
import tensorflow as tf
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("bilstm_glove_model.keras")

model = load_model()

# Load tokenizer
def load_tokenizer():
    with open("tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)
    return tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_data)

tokenizer = load_tokenizer()

# Load Label Encoder
def load_label_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)

label_encoder = load_label_encoder()
categories = list(label_encoder.classes_)

# Load test dataset for accuracy calculation
@st.cache_resource
def load_test_data():
    df = pd.read_csv("cleaned_updated_csv.csv")  # Adjust the path if needed
    X_test = df["cleaned_text"].tolist()
    y_test = label_encoder.transform(df["topic_number"])  # Ensure labels are encoded
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_padded = pad_sequences(X_test_seq, maxlen=100, padding="post")  # Adjust maxlen based on training
    return X_test_padded, y_test

X_test_padded, y_test = load_test_data()

# Compute model accuracy
predictions = model.predict(X_test_padded)
predicted_indices = np.argmax(predictions, axis=1)
accuracy = accuracy_score(y_test, predicted_indices)

st.sidebar.write(f"**Model Accuracy:** {accuracy:.2%}")

# Mapping of topic numbers to topic names
topic_mapping = {
    1.0: "Health", 2.0: "Nutrition", 3.0: "Education", 4.0: "HIV/AIDS",
    5.0: "Violence Against Children", 6.0: "WASH", 7.0: "Menstrual Hygiene",
    8.0: "Others", 9.0: "Covid", 10.0: "U-Report"
}

def preprocess_text(question):
    sequence = tokenizer.texts_to_sequences([question])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding="post")
    return padded_sequence

def classify_question(question):
    try:
        processed_text = preprocess_text(question)
        prediction = model.predict(processed_text)
        category_index = np.argmax(prediction)
        predicted_category_number = categories[category_index]
        predicted_topic_name = topic_mapping.get(predicted_category_number, "Unknown")
        return predicted_topic_name
    except Exception as e:
        return f"Error in prediction: {e}"

st.title("Swahili Question Classifier")
st.write("Andika Swali lako kwa Kiswahili.")

st.sidebar.title("Maswali")
if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Andika Swali lako:")
if st.button("Classify"):
    if question.strip():
        category = classify_question(question)
        st.session_state.history.append({"question": question, "category": category})
        st.success(f"Mada: {category}")
    else:
        st.warning("Tafadhari, Andika Swali lako.")

for item in reversed(st.session_state.history):
    st.sidebar.write(f"**Swali:** {item['question']}")
    st.sidebar.write(f"**Mada:** {item['category']}")
    st.sidebar.write("---")
