import torch
from transformers import GPT2LMHeadModel
from miditok import MMM, TokenizerConfig
import os
from settings import MODEL_DIR, GENERATE_MODEL_NAME

BOS_TOKEN = "BOS_None"
EOS_TOKEN = "Track_End"

def initialize_generate_model():
    model_path = os.path.join(MODEL_DIR, GENERATE_MODEL_NAME)
    tokenizer_path = os.path.join(MODEL_DIR, GENERATE_MODEL_NAME+'/tokenizer.json')
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = MMM(TokenizerConfig(), tokenizer_path)
    return model, tokenizer

def generate_additional_track(input_ids, model, tokenizer, temperature=0.8):
    eos_token_id = tokenizer[EOS_TOKEN]
    generated_ids = model.generate(
        input_ids,
        max_length=1024,
        do_sample=True,
        temperature=temperature,
        eos_token_id=eos_token_id,
    ).cpu()
    return generated_ids

def generate_initial_track(model, tokenizer, temperature=0.8):
    initial_token_id = tokenizer[BOS_TOKEN]
    input_ids = torch.tensor([[initial_token_id]])
    return generate_additional_track(input_ids, model, tokenizer, temperature)