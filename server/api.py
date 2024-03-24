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

from utils.generateModel_util import initialize_generate_model, generate_initial_track, generate_update_track
from utils.frontModel_util import initialize_front_model, extract_condition
from utils.utils import clear_huggingface_cache, clear_folder, extract_tempo, modify_tempo, extract_tempo
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
    logging.info(f"생성완료 : {file_path}")

    with open(file_path, 'rb') as file:
        # file_content = b64encode(file.read())
        file_content = b64encode(file.read()).decode('utf-8')
      
    # return file_content
    return JSONResponse(content={"file_content": file_content, "condition": condition})

    response = {
        "success": True,
        "content" : {"file_content": file_content, "condition": condition}
    }

    return JSONResponse(content={"response": response})

@router.post("/upload_midi/")
async def receive_midi(req: Request, request_json: UploadData):
    client_ip = req.client.host
    logging.info(f"req : {client_ip}")

    parsed_json = json.loads(request_json.request_json)
    encoded_midi = parsed_json['midi']
    instnum = parsed_json['instnum']

    # 생성 condition 저장 변수들
    emotion = parsed_json['emotion']
    tempo = parsed_json['tempo']
    genre = parsed_json['genre']

    logging.info(f"instnum : {instnum}, emotion: {emotion}, tempo: {tempo}, genre: {genre}")
    decoded_midi = b64decode(encoded_midi)

    recv_file_path = os.path.join(TEMP_DIR, client_ip.replace(".", "_") + "_recv.mid")
    try:
        # 업로드된 파일을 임시 폴더에 저장
        with open(recv_file_path, "wb") as temp_file:
            temp_file.write(decoded_midi)
        midi = Score(recv_file_path)
    except Exception as e:
        return {"status": "failed", "message": str(e)}
    
    # update midi
    midi_data = generate_update_track(generate_model, generate_tokenizer, midi, instnum, temperature=0.8)
    logging.info(f"=== {type(midi_data)}")
    if not isinstance(midi_data, Score):
        response = {
            "success": False,
            "error": "Too many tokens provided as input"
        }
        return JSONResponse(content={"response": response})
    else:
        response_status = 'failure'
        error_message = midi_data

    add_file_path = os.path.join(TEMP_DIR, client_ip.replace(".", "_") + "_add.mid")
    midi_data.dump_midi(add_file_path)

    # modify tempo
    temp_bpm = extract_tempo(recv_file_path)
    logging.info(f"before_temp_bpm : {temp_bpm}")
    if temp_bpm != None:
        modify_tempo(add_file_path, temp_bpm)
        temp_bpm = extract_tempo(add_file_path)
        logging.info(f"after_temp_bpm : {temp_bpm}")

    with open(add_file_path, 'rb') as file:
        file_content = b64encode(file.read())
    
    return file_content

    response = {
        "success": True,
        "content" : {"file_content": file_content}
    }

    return JSONResponse(content={"response": response})
