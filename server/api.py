import torch
from fastapi import APIRouter, Form, File, UploadFile, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from base64 import b64encode, b64decode
import json

from symusic import Score
from symusic.core import TempoTick
import googletrans
import os, shutil
import logging
# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from utils.generateModel_util import (
    initialize_generate_model, 
    generate_initial_track, 
    generate_updated_midi, 
    generate_update_8bar_track,
    generate_add_8bar_track
)
from utils.frontModel_util import initialize_front_model, extract_condition
from utils.anticipationModel import initialize_anticipation_model, extend_4bar_to_8bar, infill_at
from utils.utils import clear_huggingface_cache, clear_folder, extract_tempo, modify_tempo, extract_tempo, adjust_ticks_per_beat
from utils.data_processing import generate_tempo
from settings import TEMP_DIR

# 캐쉬 삭제
clear_huggingface_cache(False)

# temp 폴더 비우기
clear_folder(TEMP_DIR)

class TextData(BaseModel):
    prompt: str

class UploadData(BaseModel):
    request_json: str

translator = googletrans.Translator()

router = APIRouter()

# front model initialize
front_model, front_tokenizer = initialize_front_model()

# generate model initialize
generate_model, generate_tokenizer = initialize_generate_model()

# anticipation model initialize
anticipation_model = initialize_anticipation_model()

@router.get("/get_count/")
def get_count(req: Request):
    pass

@router.post("/generate_midi/")
async def generate_midi(req: Request, text_data: TextData):
    """
    client fetch format
    fetch(
      //  "http://0.0.0.0:8200/generate_midi/",
      "http://223.130.162.67:8200/generate_midi/",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: text,
        }),
      }
    )
    """
    client_ip = req.client.host
    logging.info(f"client_ip : {client_ip}")
    
    text = text_data.prompt
    logging.info(f"input_text : {text}")

    if text != "":
        text = translator.translate(text, dest='en').text
        logging.info(f"translate_text : {text}")

    # front model - condition 추출
    condition = extract_condition(text, front_model, front_tokenizer)
    logging.info("emotion : %s,  tempo : %s,  genre : %s", *condition)
    
    # generate model - midi track 생성
    midi_data = generate_initial_track(generate_model, generate_tokenizer, condition, top_tracks=5, temperature=0.8)
    
    # modify tempo
    temp_bpm = generate_tempo(text, condition)
    new_tempo_tick = TempoTick(time=0, qpm=temp_bpm)
    midi_data.tempos[0] = new_tempo_tick

    file_path = os.path.join(TEMP_DIR, client_ip.replace(".", "_") + "_gen.mid")
    midi_data.dump_midi(file_path)

    # modify ticks per beat
    adjust_ticks_per_beat(file_path, 200)

    logging.info(f"generate_midi : {Score(file_path)}")

    with open(file_path, 'rb') as file:
        # file_content = b64encode(file.read())
        file_content = b64encode(file.read()).decode('utf-8')
      
    # return file_content
    return JSONResponse(content={"file_content": file_content, "condition": condition})

@router.post("/upload_midi/")
async def receive_midi(req: Request, request_json: UploadData):
    client_ip = req.client.host
    logging.info(f"req : {client_ip}")

    parsed_json = json.loads(request_json.request_json)
    encoded_midi = parsed_json['midi']
    instnum = parsed_json['instnum']
    regenPart = parsed_json['regenPart']

    # 'emotion', 'tempo', 'genre' 키가 있는지 확인하고 없을 경우 기본값 할당
    emotion = parsed_json.get('emotion', 'love')
    tempo = parsed_json.get('tempo', 'Allegro')
    genre = parsed_json.get('genre', 'Pop')

    decoded_midi = b64decode(encoded_midi)

    # 업로드된 파일을 임시 폴더에 저장
    try:
        recv_file_path = os.path.join(TEMP_DIR, client_ip.replace(".", "_") + "_recv.mid")
        with open(recv_file_path, "wb") as temp_file:
            temp_file.write(decoded_midi)
        midi = Score(recv_file_path)
    except Exception as e:
        response = {
            "success": False,
            "error": str(e)
        }
        return JSONResponse(content={"response": response})
    logging.info(f"regenPart : {regenPart}, instnum : {instnum}, emotion: {emotion}, tempo: {tempo}, genre: {genre}")
    
    # update midi
    if regenPart == "default":
        midi_data = generate_updated_midi(generate_model, generate_tokenizer, midi, instnum, temperature=0.8)
    elif regenPart == "front" or regenPart == "back" :
        midi_data = generate_update_8bar_track(generate_model, generate_tokenizer, midi, instnum, regenPart, temperature=0.8)
    elif regenPart == "both":
        midi_data = generate_add_8bar_track(generate_model, generate_tokenizer, midi, instnum, temperature=0.8)

    add_file_path = os.path.join(TEMP_DIR, client_ip.replace(".", "_") + "_add.mid")
    midi_data.dump_midi(add_file_path)

    # modify ticks per beat
    adjust_ticks_per_beat(add_file_path, 200)

    # mspq, qpm 헤더 정보 저장
    from utils.anticipationModel import extract_midi_info
    mspq, qpm = extract_midi_info(recv_file_path)

    # 트랙 헤더 정보 입력
    midi_data = Score.from_file(add_file_path)
    midi_data.tempos[0].qpm = qpm
    midi_data.tempos[0].mspq = mspq
    midi_data.dump_midi(add_file_path)

    # modify tempo
    temp_bpm = extract_tempo(recv_file_path)
    if temp_bpm != None:
        modify_tempo(add_file_path, temp_bpm)
        temp_bpm = extract_tempo(add_file_path)

    logging.info(f"upload_midi : {Score(add_file_path)}")

    with open(add_file_path, 'rb') as file:
        file_content = b64encode(file.read())
    
    return file_content

# Extension : 4마디 -> 8마디 연장 (model3)
@router.post("/extend_midi/")
async def extend_midi(req: Request, request_json: UploadData):
    client_ip = req.client.host
    logging.info(f"req : {client_ip}")
    
    parsed_json = json.loads(request_json.request_json)
    encoded_midi = parsed_json['midi']
    decoded_midi = b64decode(encoded_midi)
    
    extd_recv_file_path = os.path.join(TEMP_DIR, client_ip.replace(".", "_") + "_extd_recv.mid")
    try:
        # 업로드된 파일을 임시 폴더에 저장
        with open(extd_recv_file_path, "wb") as temp_file:
            temp_file.write(decoded_midi)
        extd_recv_midi = Score(extd_recv_file_path)
    except Exception as e:
        return {"status": "failed", "message": str(e)}

    # 4마디 -> 8마디 연장
    extended_midi = extend_4bar_to_8bar(anticipation_model, extd_recv_file_path)
    extd_result_path = os.path.join(TEMP_DIR, client_ip.replace(".", "_") + "_extd_result.mid")
    extended_midi.dump_midi(extd_result_path)

    # 트랙 헤더 정보 저장
    import mido
    midi_recv = mido.MidiFile(extd_recv_file_path)
    midi_header = midi_recv.tracks[0]

    # 트랙 헤더 정보 입력
    midi_result = mido.MidiFile(extd_result_path)
    midi_result.tracks.insert(0, midi_header)
    midi_result.save(extd_result_path)

    # modify ticks per beat
    adjust_ticks_per_beat(extd_result_path, 200)

    logging.info(f"extend_midi : {Score(extd_result_path)}")

    with open(extd_result_path, 'rb') as file:
        file_content = b64encode(file.read())
    
    return file_content
    

# Infill 1마디 교체 (model3)
@router.post("/infill_midi/")
async def infill_midi(req: Request, request_json: UploadData):
    client_ip = req.client.host
    logging.info(f"req : {client_ip}")

    parsed_json = json.loads(request_json.request_json)
    infill_bar_idx = int(parsed_json['regenBarIndex'])+1

    encoded_midi = parsed_json['midi']
    decoded_midi = b64decode(encoded_midi)
    total_bars = parsed_json['totalBars']
    logging.info(f"total_bars : {total_bars}")
    infill_recv_file_path = os.path.join(TEMP_DIR, client_ip.replace(".", "_") + "_infill_recv.mid")
    try:
        with open(infill_recv_file_path, "wb") as temp_file:
            temp_file.write(decoded_midi)
        infill_recv_midi = Score(infill_recv_file_path)
    except Exception as e:
        return {"status": "failed", "message": str(e)}

    # 1마디 교체
    infill_midi = infill_at(anticipation_model, infill_recv_file_path, infill_bar_idx)
    infill_result_path = os.path.join(TEMP_DIR, client_ip.replace(".", "_") + "_infill_result.mid")
    infill_midi.dump_midi(infill_result_path)

    # modify ticks per beat
    adjust_ticks_per_beat(infill_result_path, 200)

    # 트랙 헤더 정보 저장
    import mido
    midi_recv = mido.MidiFile(infill_recv_file_path)
    midi_header = midi_recv.tracks[0]

    # 트랙 헤더 정보 입력
    midi_result = mido.MidiFile(infill_result_path)
    midi_result.tracks.insert(0, midi_header)
    midi_result.save(infill_result_path)

    logging.info(f"infill_midi : {Score(infill_result_path)}")


    with open(infill_result_path, 'rb') as file:
        file_content = b64encode(file.read())
    
    return file_content