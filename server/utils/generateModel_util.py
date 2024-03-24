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

def check_track_condition(tokenizer, generated_ids, first_track_check=True):
    """
    생성된 track ids 가 옳바르게 생성되었는지 확인하는 함수
    """

    ### Start, End Token 체크
    track_start_count = (generated_ids == tokenizer.vocab['Track_Start']).sum().item()
    track_end_count = (generated_ids == tokenizer.vocab['Track_End']).sum().item()
    bar_start_count = (generated_ids == tokenizer.vocab['Bar_Start']).sum().item()
    bar_end_count = (generated_ids == tokenizer.vocab['Bar_End']).sum().item()

    # logging.info(f"track count (start, end) : {track_start_count}, {track_end_count}")
    # logging.info(f"bar count (start, end) : {bar_start_count}, {bar_end_count}")

    if track_start_count != track_end_count or bar_start_count != bar_end_count:
        logging.info("Token pairs do not match.")
        return False
    
    if (bar_start_count / track_start_count) > 4:
        logging.info("Generated track exceeds 4 bars.")
        return False
    
    if track_start_count == 1 and first_track_check==True:
        if bar_start_count != 4:
            logging.info("First generated track is not 4 bars long.")
            return False

        bar_start_first_token_index = (generated_ids == tokenizer.vocab['Bar_Start']).nonzero()[0][1].item()
        bar_start_end_token_index = (generated_ids == tokenizer.vocab['Bar_End']).nonzero()[0][1].item()
        if bar_start_first_token_index == bar_start_end_token_index-1:
            logging.info("No content generated for the first bar of the first track.")
            return False

    
    ### 중복 악기 생성 체크
    instrument_vocab_ids = [value for key, value in tokenizer.vocab.items() if 'Program_' in key]
    instrument_ids = [item.item() for item in generated_ids[0] if item.item() in instrument_vocab_ids]
    is_duplicate = len(set(instrument_ids)) != len(instrument_ids)
    # logging.info(f"instrument_ids : {instrument_ids}")
    if is_duplicate:
        logging.info(f"Duplicate instruments were created.")
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
            input_text = BOS_TOKEN + " Genre_" + genre + " Emotion_" + emotion + " Track_Start"
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

def generate_track_update(model, tokenizer, token_ids, track_num, temperature=0.8):
    if track_num == 999:
        updated_text = "Track_Start"
    else:
        updated_text = f"Track_Start Program_{track_num}"

    token_list = [tokenizer[token] for token in updated_text.split()]
    mmm_tokens_ids = token_ids + token_list
    nnn_tokens_ids = mmm_to_nnn(mmm_tokens_ids, tokenizer)
    nnn_tokens_ids = torch.tensor([nnn_tokens_ids]).to(DEVICE)
    
    for i in range(5):
        generated_ids = generate_additional_track(nnn_tokens_ids, model, tokenizer, temperature)
        if check_track_condition(tokenizer, generated_ids, first_track_check=False):
            break

    mmm_generated_ids = nnn_to_mmm(generated_ids[0].tolist(), tokenizer)
    return mmm_generated_ids

def generate_updated_midi(model, tokenizer, midi, track_num, temperature=0.8):
    token_ids = tokenizer(midi).ids
    mmm_generated_ids = generate_track_update(model, tokenizer, token_ids, track_num, temperature)
    midi_data = tokenizer.tokens_to_midi(mmm_generated_ids)
    return midi_data

def chunk_tokens_by_token(tokenizer, ids_list, music_unit):
    """
    주어진 토큰 리스트를 music_unit으로 구분하여 chunk하는 함수
    """
    track_chunks = []
    start_idx = 0
    start_token = music_unit + "_Start"
    end_token = music_unit + "_End"

    for i, token in enumerate(ids_list):
        if token == tokenizer.vocab[start_token]:
            start_idx = i
            if start_idx != 0 and not track_chunks:
                track_chunks.append(ids_list[:start_idx])
        if token == tokenizer.vocab[end_token]:
            track_chunks.append(ids_list[start_idx:i+1])
            start_idx = i+1

    if start_idx != len(ids_list):
        track_chunks.append(ids_list[start_idx:])
    
    return track_chunks

def chunk_tracks_and_bars(ids_list, tokenizer):
    """
    트랙과 바를 chunk하는 함수
    """
    track_chunks_ids = chunk_tokens_by_token(tokenizer, ids_list, "Track")
    track_bar_chunks_ids = []
    for track in track_chunks_ids:
        bar_chunk_track = chunk_tokens_by_token(tokenizer, track, "Bar")
        track_bar_chunks_ids.append(bar_chunk_track)
    return track_bar_chunks_ids

def extract_index(regenPart):
    """
    재생성 부분에 대한 인덱스 범위를 추출하는 함수
    """
    if regenPart == "front":
        return 1, 4
    elif regenPart == "back":
        return 5, 8

def extract_four_bars(track_bar_chunks_ids, regenPart):
    """
    재생성 부분에 해당하는 네 바를 추출하는 함수
    """
    regen_start_idx, regen_end_idx = extract_index(regenPart)

    four_bar_chunks_ids = []
    for track in track_bar_chunks_ids:
        four_bar_chunk = [track[0]] + track[regen_start_idx:regen_end_idx+1] + [track[-1]]
        four_bar_chunks_ids.append(four_bar_chunk)

    return four_bar_chunks_ids

def generate_update_8bar_track(model, tokenizer, midi, track_num, regenPart, temperature=0.8):
    """
    8바 트랙을 업데이트하고 재생성하는 함수
    """
    # 트랙과 바를 chunk
    all_tracks_track_bar_chunks_ids = chunk_tracks_and_bars(tokenizer(midi).ids, tokenizer)
    all_track_four_bar_chunk = extract_four_bars(all_tracks_track_bar_chunks_ids, regenPart)

    # 재생성
    regen_input_ids = [element for track in all_track_four_bar_chunk[:-1] for bar in track for element in bar]
    generated_ids = generate_track_update(model, tokenizer, regen_input_ids, track_num, temperature)

    # 업데이트
    regen_start_idx, regen_end_idx = extract_index(regenPart)
    generated_track_chunk_ids = chunk_tokens_by_token(tokenizer, generated_ids, "Track")
    generated_bars_ids = chunk_tokens_by_token(tokenizer, generated_track_chunk_ids[-1], "Bar")[1:-1]
    all_tracks_track_bar_chunks_ids[-1][regen_start_idx:regen_end_idx+1] = generated_bars_ids

    # 재생성된 토큰을 MIDI로 변환
    regen_ids = [element for track in all_tracks_track_bar_chunks_ids for bar in track for element in bar]
    midi_data = tokenizer.tokens_to_midi(regen_ids)

    return midi_data

def generate_add_8bar_track(model, tokenizer, midi, track_num, temperature=0.8):
    pass