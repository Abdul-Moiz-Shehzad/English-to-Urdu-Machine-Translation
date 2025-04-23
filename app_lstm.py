import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import joblib
from tensorflow.keras.models import load_model
import os
from src.utils.config_loader import load_config 

tokenized_english_sentences=joblib.load('./data/artifacts/tokenized-english-sentences.joblib')
tokenized_urdu_sentences=joblib.load('./data/artifacts/tokenized-urdu-sentences.joblib')
urdu_tokenizer=joblib.load('./data/artifacts/urdu-tokenizer.joblib')
english_tokenizer=joblib.load('./data/artifacts/english-tokenizer.joblib')

def translate_user_prompt(input_sentence: str,
                          model,
                          english_tokenizer: Tokenizer,
                          urdu_tokenizer: Tokenizer,
                          max_length_english: int,
                          max_length_urdu: int) -> str:
  
    input_sentence_with_tokens = f"<start> {input_sentence} <end>"
    tokenized_sentence = english_tokenizer.texts_to_sequences([input_sentence_with_tokens])
    padded_sentence = pad_sequences(tokenized_sentence, maxlen=max_length_english, padding='post', truncating='post')

    start_token_index = urdu_tokenizer.word_index['<start>']
    decoder_input = np.zeros((1, max_length_urdu - 1))  
    decoder_input[0, 0] = start_token_index

    translated_sentence = []
    for i in range(max_length_urdu - 1):
        predictions = model.predict([padded_sentence, decoder_input], verbose=0)

        predicted_word_index = np.argmax(predictions[0, i, :])

        if predicted_word_index == urdu_tokenizer.word_index['<end>']:
            break

        translated_sentence.append(predicted_word_index)

        if i + 1 < (max_length_urdu - 1):
            decoder_input[0, i + 1] = predicted_word_index

    index_to_word = {id: word for word, id in urdu_tokenizer.word_index.items()}
    index_to_word[0] = '<PAD>'  

    decoded_translation = ' '.join([index_to_word[idx] for idx in translated_sentence
                                    if idx not in [0, urdu_tokenizer.word_index['<start>'], urdu_tokenizer.word_index['<end>']]])

    return decoded_translation.strip()

lstm_model=load_model('./data/models/LSTM_seq2seq.keras')
config=load_config('./config.yaml')
max_english_length=config['Preprocessing']['Padding']['English']
max_urdu_length=config['Preprocessing']['Padding']['Urdu']
models = {'LSTM': lstm_model}


st.set_page_config(page_title="English to Urdu Translation", layout="centered")
st.title("English to Urdu Machine Translation")
st.write("Enter an English sentence to see the Urdu translation.")

user_input = st.text_input("Enter English sentence", "")

if st.button("Translate"):
    if not user_input.strip():
        st.warning("Please enter a sentence.")
    else:

        translation = translate_user_prompt(
            input_sentence=user_input,
            model=models['LSTM'],
            english_tokenizer=english_tokenizer,
            urdu_tokenizer=urdu_tokenizer,
            max_length_english=max_english_length,
            max_length_urdu=max_urdu_length
        )

        st.success(f"**Urdu Translation:** {translation}")
