import torch
from fastapi import APIRouter, Form, File, UploadFile, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from base64 import b64encode, b64decode
import json

from symusic import Score
import os, shutil
import logging
# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from utils.utils import clear_huggingface_cache
from utils.generateModel_util import initialize_generate_model, generate_initial_track, generate_update_track
from utils.frontModel_util import initialize_front_model, extract_condition
from settings import TEMP_DIR

# 캐쉬 삭제
clear_huggingface_cache(False)

class TextData(BaseModel):
    prompt: str

class Base64Request(BaseModel):
    base64_file: str

class Base64Response(BaseModel):
    response_code: str
    base64_file: str

class UploadData(BaseModel):
    request_json: str

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

    # front model - condition 추출
    condition = extract_condition(text, front_model, front_tokenizer)
    logging.info("emotion : %s,  tempo : %s,  genre : %s", *condition)
    
    # generate model - midi track 생성
    midi_data = generate_initial_track(generate_model, generate_tokenizer, condition, top_tracks=5, temperature=0.8)

    file_path = os.path.join(TEMP_DIR, client_ip.replace(".", "_") + "_gen.mid")
    midi_data.dump_midi(file_path)
    logging.info(f"생성완료 : {file_path}")

    with open(file_path, 'rb') as file:
        file_content = b64encode(file.read())
      
    return file_content

@router.post("/upload_midi/")
async def receive_midi(req: Request, request_json: UploadData):
    client_ip = req.client.host
    logging.info(f"req : {client_ip}")

    parsed_json = json.loads(request_json.request_json)
    encoded_midi = parsed_json['midi']
    instnum = parsed_json['instnum']
    decoded_midi = b64decode(encoded_midi)

    temp_file_path = os.path.join(TEMP_DIR, client_ip.replace(".", "_") + "_recv.mid")
    try:
        # 업로드된 파일을 임시 폴더에 저장
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(decoded_midi)
        midi = Score(temp_file_path)
    except Exception as e:
        return {"status": "failed", "message": str(e)}
    
    # # update midi
    midi_data = generate_update_track(generate_model, generate_tokenizer, midi, instnum, temperature=0.8)

    file_path = os.path.join(TEMP_DIR, client_ip.replace(".", "_") + "_add.mid")
    midi_data.dump_midi(file_path)

    with open(file_path, 'rb') as file:
        file_content = b64encode(file.read())
    
    return file_content


# @router.post("/upload_midi/")
# async def receive_midi(req: Request, midi_file: UploadFile = File(...), instnum: int = Form(...)):
#     client_ip = req.client.host
#     temp_file_path = os.path.join(TEMP_DIR, client_ip.replace(".", "_") + "_recv.mid")
#     try:
#         file_content = b64decode(midi_file.base64_file)
#         base64_file = file_content.decode('utf-8')
#         # 업로드된 파일을 임시 폴더에 저장
#         with open(temp_file_path, "wb") as temp_file:
#             # shutil.copyfileobj(midi_file.file, temp_file)
#             temp_file.write(base64_file)

#         midi = Score(temp_file_path)
#     except Exception as e:
#         return {"status": "failed", "message": str(e)}
    
#     # update midi
#     midi_data = generate_update_track(generate_model, generate_tokenizer, midi, instnum, temperature=0.8)

#     file_path = os.path.join(TEMP_DIR, client_ip.replace(".", "_") + "_add.mid")
#     midi_data.dump_midi(file_path)

#     return FileResponse(file_path, media_type="audio/midi")
