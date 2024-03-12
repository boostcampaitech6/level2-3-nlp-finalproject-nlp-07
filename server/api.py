from fastapi import APIRouter, Form, File, UploadFile, Request
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from transformers import GPT2LMHeadModel
from miditok import MMM, TokenizerConfig
import torch
import os, shutil
from symusic import Score
import logging
# 로깅 설정
logging.basicConfig(level=logging.INFO)

from pydantic import BaseModel
from utils.generate import generate_initial_track, generate_additional_track

class TextData(BaseModel):
    prompt: str

router = APIRouter()
templates = Jinja2Templates(directory="templates")

MODEL_DIR = "./model/"
TEMP_DIR = "./temp/"

MODEL_NAME = "bar4-ch4-checkpoint-8100"

model_path = os.path.join(MODEL_DIR, MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(model_path) 

tokenizer_path = os.path.join(MODEL_DIR, MODEL_NAME+'/tokenizer.json')
tokenizer = MMM(TokenizerConfig(), tokenizer_path)

@router.post("/generate_midi/")
async def generate_midi(req: TextData):
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
    text = req.prompt
    logging.info(f"input_text : {text}")
    
    ## generation midi
    generated_ids = generate_initial_track(model, tokenizer, temperature=0.8)

    midi_data = tokenizer.tokens_to_midi(generated_ids[0])

    file_path = os.path.join(TEMP_DIR, "temp_gen.mid")
    midi_data.dump_midi(file_path)

    return FileResponse(file_path, media_type="audio/midi")
    return StreamingResponse(open(file_path, "rb"), media_type="audio/midi")

@router.post("/upload_midi/")
async def receive_midi(midi_file: UploadFile = File(...), instnum: int = Form(...)) -> str:
    temp_file_path = os.path.join(TEMP_DIR, "temp_upload.mid")
    try:
        # 업로드된 파일을 임시 폴더에 저장
        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(midi_file.file, temp_file)
        
        # 임시 파일 경로 반환
        return {"status": "success", "temp_file_path": temp_file_path}
    except Exception as e:
        return {"status": "failed", "message": str(e)}