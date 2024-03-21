import torch
from transformers import GPT2LMHeadModel
from miditok import MMM, TokenizerConfig
import os
from settings import MODEL_DIR, GENERATE_MODEL_NAME
from utils.data_processing import get_instruments_for_generate_model
from utils.tokenizer_converter import mmm_to_nnn, nnn_to_mmm
from tokenizer_svr import get_nnn_tokenizer, get_nnn_meta_tokenizer
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BOS_TOKEN = "BOS_None"
EOS_TOKEN = "Track_End"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def initialize_generate_model():
    # model_path = os.path.join(MODEL_DIR, GENERATE_MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(GENERATE_MODEL_NAME).to(DEVICE)

    tokenizer = get_nnn_meta_tokenizer(4)
    model.resize_token_embeddings(len(tokenizer))
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

    return generated_ids

def check_track_condition(tokenizer, generated_ids):
    """
    생성된 track ids 가 옳바르게 생성되었는지 확인하는 함수
    """

    track_start_count = (generated_ids == tokenizer.vocab['Track_Start']).sum().item()
    track_end_count = (generated_ids == tokenizer.vocab['Track_End']).sum().item()
    bar_start_count = (generated_ids == tokenizer.vocab['Bar_Start']).sum().item()
    bar_end_count = (generated_ids == tokenizer.vocab['Bar_End']).sum().item()

    logging.info(f"track count (start, end) : {track_start_count}, {track_end_count}")
    logging.info(f"bar count (start, end) : {bar_start_count}, {bar_end_count}")

    if track_start_count != track_end_count or bar_start_count != bar_end_count:
        logging.info("Token pairs do not match.")
        return False
    
    if track_start_count == 1 and bar_start_count != 4:
        logging.info("First generated track is not 4 bars long.")
        return False
    
    if (bar_start_count / track_start_count) > 4:
        logging.info("Generated track exceeds 4 bars.")
        return False

    return True 

def generate_initial_track(model, tokenizer, condition, top_tracks=5, temperature=0.8):
    emotion, tempo, genre = condition
    track_counter  = 0
    # while track_counter < 4:
    for i in range(10):
        logging.info("-"*5 + f"track_counter : {track_counter}" + "-"*5)
        if i == 0:
            logging.info(f"track count (start, end)")
            input_text = BOS_TOKEN + " Genre_" + genre + " Emotion_" + emotion
            # input_text = BOS_TOKEN + " Genre_" + genre + " Emotion_" + emotion + " Tempo_" + tempo
            current_track_ids = torch.empty(1, 0).to(DEVICE)
            logging.info(f"input : {input_text}")

        token_list = [tokenizer[token] for token in input_text.split()]
        token_ids = torch.tensor([token_list]).to(DEVICE)
        input_ids = torch.cat((current_track_ids, token_ids), dim=1).to(torch.int64)

        generated_ids = generate_additional_track(input_ids, model, tokenizer, temperature)

        if check_track_condition(tokenizer, generated_ids):
            current_track_ids = generated_ids
            track_counter += 1

        if track_counter >= 4:
            break
    
    mmm_tokens_ids = nnn_to_mmm(generated_ids[0].tolist(), tokenizer)
    # logging.info(mmm_tokens_ids)
    midi_data = tokenizer.tokens_to_midi(mmm_tokens_ids)
    return midi_data

def generate_update_track(model, tokenizer, midi, track_num, temperature=0.8):
    if track_num == 999:
        updated_text = "Track_Start"
    else:
        updated_text = f"Track_Start Program_{track_num}"

    token_list = [tokenizer[token] for token in updated_text.split()]
    mmm_tokens_ids = tokenizer(midi).ids + token_list
    nnn_tokens_ids = mmm_to_nnn(mmm_tokens_ids, tokenizer)
    nnn_tokens_ids = torch.tensor([nnn_tokens_ids]).to(DEVICE)
    
    generated_ids = generate_additional_track(nnn_tokens_ids, model, tokenizer, temperature)
    mmm_generated_ids = nnn_to_mmm(generated_ids[0].tolist(), tokenizer)
    midi_data = tokenizer.tokens_to_midi(mmm_generated_ids)

    return midi_data