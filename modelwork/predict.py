import pickle
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def load_model(pickle_file_path):
    with open(pickle_file_path, 'rb') as file:
        model = joblib.load(file)
    return model


def load_vectorizer(pickle_file_path):
    with open(pickle_file_path, 'rb') as file:
        vectorizer = pickle.load(file)
    return vectorizer


def predict(processed_text, model, vectorizer):
    vectorized_text = vectorizer.transform([processed_text])
    result = model.predict(vectorized_text)
    return result
