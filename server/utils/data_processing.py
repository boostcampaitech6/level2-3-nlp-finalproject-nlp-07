import os
import pickle
import random
from settings import PICKLE_DIR, GENRE_INSTRUMENTS_PICKLE_PATH

def get_instruments_for_generate_model(condition):
    emotion, tempo, genre = condition
    genre_instruments_path = os.path.join(PICKLE_DIR, GENRE_INSTRUMENTS_PICKLE_PATH)
    with open(genre_instruments_path, 'rb') as f:
        loaded_genre_instruments = pickle.load(f)

    if genre in loaded_genre_instruments:
        genre_instruments = loaded_genre_instruments[genre]
    else:
        raise ValueError(f"Genre '{genre}' is not found in the data.")

    return genre_instruments

def simple_hash(text):
    # 해시값 초기화
    hash_value = 0
    
    # 각 문자에 대한 ASCII 코드를 이용하여 해시값 계산
    for char in text:
        hash_value = (hash_value * 31 + ord(char)) % 2**32
    
    return hash_value

def generate_tempo(prompt,condition):
    emotion,tempo,genre = condition
    random.seed(simple_hash(prompt+emotion+tempo+genre))
    if tempo == "Presto":
        return random.randint(144, 172)
    elif tempo == "Allegro":
        return random.randint(116, 143)
    elif tempo == "Moderato":
        return random.randint(80, 115)
    elif tempo == "Andante":
        return random.randint(44, 79)
    else:
        return "Invalid token"