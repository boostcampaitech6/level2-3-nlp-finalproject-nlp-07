import os
import json

def load_config(config_file):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, config_file)
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

config = load_config('config.json')

MODEL_DIR = config['MODEL_DIR']
GENERATE_MODEL_NAME = config['GENERATE_MODEL_NAME']
FRONT_MODEL_NAME = config['FRONT_MODEL_NAME']   # hugging face
PICKLE_PATH = config['PICKLE_PATH']
GENRE_INSTRUMENTS_PICKLE_PATH = config['GENRE_INSTRUMENTS_PICKLE_PATH']
TEMP_DIR = config['TEMP_DIR']
LOGS_DIR = config['LOGS_DIR']


create_folder_if_not_exists(MODEL_DIR)
create_folder_if_not_exists(TEMP_DIR)