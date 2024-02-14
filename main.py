import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from PIL import Image

def load_model_and_vectorizer(model_path, vectorizer_path):
    with open(model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    
    return loaded_model, vectorizer

def preprocess_and_vectorize(text, vectorizer):
    vectorized_text = vectorizer.transform([text])
    return vectorized_text

def predict_disaster(model, vectorized_text):
    prediction = model.predict(vectorized_text)[0]
    return prediction

def main():
    model_path = 'disaster_model.pkl'
    vectorizer_path = 'tfidf_vectorizer.pkl'
    loaded_model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)

    st.markdown(
        """
        <style>
        body {
            background-color: #f5f5f5; /* Set your desired background color */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('Twitter Disaster Detection')
    st.write("This app analyzes a tweet and predicts whether it's about a disaster or a normal situation.")

    user_input = st.text_area('Enter a tweet:')
    
    if st.button('Analyze Tweet'):
        if user_input:
            vectorized_text = preprocess_and_vectorize(user_input, vectorizer)
            prediction = predict_disaster(loaded_model, vectorized_text)
            disaster_mapping = {1: 'Disaster 🚨', 0: 'Normal 🌤️'}
            disaster_result = disaster_mapping.get(prediction, 'Unknown')
            st.write(f"Tweet Type: {disaster_result}")

    image_path = 'background.png'  
    image = Image.open(image_path)
    st.image(image, use_column_width=True)

if __name__ == "__main__":
    main()
