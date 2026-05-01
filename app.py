import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import TweetTokenizer
import streamlit as st
import pickle

max_len = 13

# ----------------------
# Functions
# ----------------------

def preprocess(text):
    pre_text = [word for word in TweetTokenizer().tokenize(text.lower()) if word.isalpha() or len(word) > 1]
    return " ".join(pre_text)


def pred_sent(sent, model, tokenizer):
    sent_pre = preprocess(sent)  # preprocess the sentence using the same preprocessing function as training data
    
    # convert text to sequences of integers based on the tokenizer
    sent_seq = tokenizer.texts_to_sequences([sent_pre])  # tokenizer expects a list of sentences

    # pad sequences to ensure uniform input length
    sent_pad = pad_sequences(sent_seq, maxlen=max_len, padding='post')

    pred = model.predict(sent_pad).argmax(axis=-1)[0]

    label_to_token = {
        0: "PAD",
        1: "EN",
        2: "FR"
    }

    result = {}
    for i, token in enumerate(sent_pre.split()):
        if i >= max_len:  # if sentence is longer than max_len, ignore extra tokens
            break
        result[token] = label_to_token[pred[i]]

    return result

# ----------------------
# Load model + tokenizer
# ----------------------
model = tf.keras.models.load_model("language_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ----------------------
# UI
# ----------------------
st.title("Code-Mixed Classifier (EN / FR)")

st.divider()

st.write("by:")
st.write("Omar Nezar Jaber Jaber -16s2135907")
st.write("MOHAMMED HAMED SAID AL OUFI -56s2197")
st.write("IBRAHIM HAMED JUMA AL-SHAIBANI -16j2124137")

st.divider()

user_input = st.text_area("Enter a sentence:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter text")
    else:
        if(len(user_input.strip().split()) > max_len):
            st.warning(f"Some tokens may be ignored if you enter a sentence with more than {max_len} tokens")
        result = pred_sent(
            user_input,
            model,
            tokenizer,
        )

        st.write("Prediction per token:")
        st.write(result)