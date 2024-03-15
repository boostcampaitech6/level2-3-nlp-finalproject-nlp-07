import torch
from transformers import GPT2LMHeadModel
from miditok import MMM, TokenizerConfig
import os
from settings import MODEL_DIR, GENERATE_MODEL_NAME
from utils.data_processing import get_instruments_for_generate_model
from utils.tokenizer_converter import mmm_to_nnn, nnn_to_mmm
from tokenizer_svr import get_nnn_tokenizer

BOS_TOKEN = "BOS_None"
EOS_TOKEN = "Track_End"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def initialize_generate_model():
    model_path = os.path.join(MODEL_DIR, GENERATE_MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(DEVICE)

    # tokenizer_path = os.path.join(MODEL_DIR, GENERATE_MODEL_NAME+'/tokenizer.json')
    # tokenizer = MMM(TokenizerConfig(), tokenizer_path)
    tokenizer = get_nnn_tokenizer(4)
    return model, tokenizer

def generate_additional_track(input_ids, model, tokenizer, temperature=0.8):
    eos_token_id = tokenizer[EOS_TOKEN]
    generated_ids = model.generate(
        input_ids.to(DEVICE),
        max_length=1024,
        do_sample=True,
        temperature=temperature,
        eos_token_id=eos_token_id,
    )
    # ).cpu()
    return generated_ids

def generate_initial_track(model, tokenizer, condition, track_num=5, temperature=0.8):
    genre_instruments = get_instruments_for_generate_model(condition)[:track_num]
    for i, instruments in enumerate(genre_instruments):
        if i == 0:
            input_text = "BOS_None Track_Start " + instruments
            generated_ids = torch.empty(1, 0).to(DEVICE)
        else:
            input_text = "Track_Start " + instruments

        token_list = [tokenizer[token] for token in input_text.split()]
        token_ids = torch.tensor([token_list]).to(DEVICE)
        input_ids = torch.cat((generated_ids, token_ids), dim=1).to(torch.int64)

        generated_ids = generate_additional_track(input_ids, model, tokenizer, temperature)
    
    mmm_tokens_ids = nnn_to_mmm(generated_ids[0].tolist(), tokenizer)
    midi_data = tokenizer.tokens_to_midi(mmm_tokens_ids)

    return midi_data