import pickle
import string
import streamlit as st
import re

# NO NLTK IMPORTS!

ps = None  # We'll handle stemming differently


def simple_tokenize(text):
    """Simple tokenizer that doesn't need NLTK"""
    # Convert to lowercase
    text = text.lower()
    # Split on non-alphanumeric characters
    words = re.findall(r'\b\w+\b', text)
    return words


def simple_stem(word):
    """Very simple stemmer (just removes common endings)"""
    if word.endswith('ing'):
        return word[:-3]
    if word.endswith('ed'):
        return word[:-2]
    if word.endswith('s') and len(word) > 3:
        return word[:-1]
    return word


def transform_text(text):
    # Simple tokenization
    words = simple_tokenize(text)

    # Simple stopwords list
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
                  'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                  'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
                  'may', 'might', 'must', 'can', 'could', 'of', 'for', 'to', 'in',
                  'on', 'at', 'by', 'with', 'from', 'up', 'down', 'out', 'over',
                  'under', 'again', 'further', 'then', 'once'}

    # Filter and stem
    filtered_words = []
    for word in words:
        if word not in stop_words and word not in string.punctuation:
            filtered_words.append(simple_stem(word))

    return " ".join(filtered_words)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")
input_email = st.text_input("Enter the message")

if st.button("Predict"):
    transformed_email = transform_text(input_email)
    vector_input = tfidf.transform([transformed_email])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")