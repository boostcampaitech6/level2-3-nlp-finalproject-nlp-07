import os
import pickle
from settings import MODEL_DIR, GENRE_INSTRUMENTS_PICKLE_PATH

def get_instruments_for_generate_model(condition):
    emotion, tempo, genre = condition
    genre_instruments_path = os.path.join(MODEL_DIR, GENRE_INSTRUMENTS_PICKLE_PATH)
    with open(genre_instruments_path, 'rb') as f:
        loaded_genre_instruments = pickle.load(f)

    if genre in loaded_genre_instruments:
        genre_instruments = loaded_genre_instruments[genre]
    else:
        raise ValueError(f"Genre '{genre}' is not found in the data.")

    return genre_instruments