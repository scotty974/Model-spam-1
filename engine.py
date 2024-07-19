import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def load_model(model_path='model.pkl'):
    """
    Load a model from a file.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        print('Model loaded')
    return model

def load_vectorizer(vectorizer_path='vectorizer.pkl'):
    """
    Load a vectorizer from a file.
    """
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
        print('Vectorizer loaded')
    return vectorizer

def preprocess_text(text, vectorizer):
    """
    Preprocess the input text using the provided vectorizer.
    """
    return vectorizer.transform([text])

def evaluate_text(model, vectorizer, text):
    """
    Evaluate a single piece of text to determine if it's spam.
    """
    preprocessed_text = preprocess_text(text, vectorizer)
    prediction = model.predict(preprocessed_text)
    return prediction

if __name__ == "__main__":
    model = load_model('model.pkl')
    vectorizer = load_vectorizer('vectorizer.pkl')
    sample_text = "Congratulations! You've won a free ticket to the Bahamas. Call now to claim!"
    result = evaluate_text(model, vectorizer, sample_text)
    print('Spam' if result[0] == 1 else 'Not Spam')
