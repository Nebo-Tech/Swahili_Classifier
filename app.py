import streamlit as st
import tensorflow as tf
import numpy as np
import json

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("bilstm_glove_new_model.keras")

model = load_model()

# Define categories (must match training labels)

# Mapping of topic numbers to topic names
topic_mapping = {
    1.0: "Health", 2.0: "Nutrition", 3.0: "Education", 4.0: "HIV/AIDS",
    5.0: "Violence Against Children", 6.0: "WASH", 7.0: "Menstrual Hygiene",
    8.0: "Others", 9.0: "Covid", 10.0: "U-Report"
}


# Function to preprocess input
def preprocess_text(question, tokenizer_path="tokenizer.json", max_length=100):
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)
    
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_data)
    
    sequence = tokenizer.texts_to_sequences([question])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_length, padding="post")
    return padded_sequence

# Function to classify question
def classify_question(question):
    try:
        processed_text = preprocess_text(question)
        prediction = model.predict(processed_text)
        category_index = np.argmax(prediction)  # Get index of highest probability
        
        # Map the predicted index to the corresponding category
        return topic_mapping.get(category_index + 1.0, "Unknown Category")
    except Exception as e:
        return f"Error in prediction: {e}"

# Streamlit UI
st.title("Swahili Question Classifier")
st.write("Type a question in Swahili and get its classification.")

# Sidebar for question history
st.sidebar.title("History")
if "history" not in st.session_state:
    st.session_state.history = []

# User input
question = st.text_input("Andika Swali lako:")
if st.button("Classify"):
    if question.strip():
        category = classify_question(question)
        st.session_state.history.append({"question": question, "category": category})
        st.success(f"Mada: {category}")
    else:
        st.warning("Tafadhari, Andika Swali lako.")

# Display question history in the sidebar
for item in reversed(st.session_state.history):
    st.sidebar.write(f"**Swali:** {item['question']}")
    st.sidebar.write(f"**Mada:** {item['category']}")
    st.sidebar.write("---")
