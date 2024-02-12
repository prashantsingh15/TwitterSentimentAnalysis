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


def predict_sentiment(model, vectorized_text):
    
    prediction = model.predict(vectorized_text)[0]
    return prediction


def main():
   
    model_path = 'sentiment_model.pkl'
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

   
    st.title('Twitter Sentiment Analysis')
    st.write("This app analyzes the sentiment of a tweet and predicts whether it's positive, neutral, or negative.")

   
    user_input = st.text_area('Enter a tweet:')
    
    if st.button('Analyze Sentiment'):
        if user_input:
            
            vectorized_text = preprocess_and_vectorize(user_input, vectorizer)

            
            prediction = predict_sentiment(loaded_model, vectorized_text)

            
            sentiment_mapping = {1: 'Positive 😄', 0: 'Neutral 😐', -1: 'Negative 😞'}
            sentiment_result = sentiment_mapping.get(prediction, 'Unknown')
            
            st.write(f"Sentiment: {sentiment_result}")

   
    image_path = 'background.png'  
    image = Image.open(image_path)
    st.image(image, use_column_width=True)

if __name__ == "__main__":
    main()
