__author__ = 'Alon'
import pickle

def save_model(model, filename):
    with open(filename, 'wb+') as f:
        pickle.dump(model, f)

def load_model(filename):
    try:
        with open(filename, 'rb+') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
