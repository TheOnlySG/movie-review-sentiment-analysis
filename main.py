from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import streamlit as st

model = load_model('model/simpleRNN.h5')
word_index = imdb.get_word_index()
reversed_word_index = {value : key for key , value
                       in word_index.items()}

def preprocess_text(text):
    words = text.lower().split() #not this is a list

    encoded_review = [word_index.get(word , 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review] , maxlen=500)
    return padded_review


def predict_sentiment(review):
    preprocess_input = preprocess_text(review)
    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment , prediction[0][0]


st.title('IMDB Movie Review Sentiment Analysis')
st.subheader('Enter a movie review to classify positive or negative.')


user_input = st.text_area('Movie Review')

if st.button('Classify'):
    # preprocessed_input = preprocess_text(user_input)
    sentiment , prediction = predict_sentiment(user_input)

    st.write(f'Sentiment  : {sentiment}')
    st.write(f'prediction : {prediction}')
else :
    st.write(f'please write movie review')

