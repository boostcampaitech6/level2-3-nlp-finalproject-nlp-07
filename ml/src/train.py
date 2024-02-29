# from transformers import GPT2LMHeadModel
from tokenizer import get_custom_tokenizer
from load_data import load_midi_paths

def main():
    MOEL_NAME = "gpt2"
    
    tokenizer = get_custom_tokenizer()
    
    train_paths = ['../data/chunks/']
    train_midi_paths = load_midi_paths(train_paths)

if __name__ == '__main__':
    print('Training model...')
    main()