import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

VECTORIZER_PATH = 'models/vectorizer.pkl'

def create_vectorizer(type='tfidf', max_features=5000):
    """
    Creates and returns a new vectorizer instance.
    """
    if type == 'bow':
        return CountVectorizer(max_features=max_features, ngram_range=(1, 2))
    else:
        return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))

def save_vectorizer(vectorizer, path=VECTORIZER_PATH):
    """
    Saves the trained vectorizer to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(vectorizer, f)

def load_vectorizer(path=VECTORIZER_PATH):
    """
    Loads a vectorizer from disk.
    """
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

def fit_transform_text(texts, vectorizer_type='tfidf'):
    """
    Fits a vectorizer on texts and returns the matrix + vectorizer.
    """
    vectorizer = create_vectorizer(vectorizer_type)
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer

def transform_text(texts, vectorizer):
    """
    Transforms texts using an existing vectorizer.
    """
    return vectorizer.transform(texts)
