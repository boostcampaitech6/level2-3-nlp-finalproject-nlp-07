from fastapi import APIRouter, Form, File, UploadFile, Request
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from transformers import GPT2LMHeadModel
from tokenizer import get_custom_tokenizer
import torch
import os, shutil
from symusic import Score
import logging
# 로깅 설정
logging.basicConfig(level=logging.INFO)

from pydantic import BaseModel

class TextData(BaseModel):
    text: str

router = APIRouter()
templates = Jinja2Templates(directory="templates")

MODEL_DIR = "./model/"
TEMP_DIR = "./temp/"

model_path = os.path.join(MODEL_DIR, "bar4-ch4-checkpoint-8100")
model = GPT2LMHeadModel.from_pretrained(model_path) 
tokenizer = get_custom_tokenizer()

@router.post("/generate_midi/")
async def generate_midi(req: Request):
    """
    client fetch format
    fetch(
      "http://0.0.0.0:8200/generate_midi/",
      // "http://223.130.130.56:8200/generate_midi/",
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
    data = await req.json()
    text = data["prompt"]
    logging.info(f"Received text: {text}")
    
    ## generation midi
    initial_token = "BOS_None"
    generated_ids = torch.tensor([[tokenizer[initial_token]]])

    iteration_number = 0

    input_ids = generated_ids
    eos_token_id = tokenizer["Track_End"]
    temperature = 0.8
    generated_ids = model.generate(
        input_ids,
        max_length=1024,
        do_sample=True,
        temperature=temperature,
        eos_token_id=eos_token_id,
    ).cpu()

    midi_data = tokenizer.tokens_to_midi(generated_ids[0])

    file_path = os.path.join(TEMP_DIR, "temp_gen.mid")
    midi_data.dump_midi(file_path)

    return FileResponse(file_path, media_type="audio/midi")
    return StreamingResponse(open(file_path, "rb"), media_type="audio/midi")

@router.post("/upload_midi/")
async def upload_midi(midi_file: UploadFile = File(...)):
    temp_file_path = os.path.join(TEMP_DIR, "temp_upload.mid")
    try:
        # 업로드된 파일을 임시 폴더에 저장
        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(midi_file.file, temp_file)
        
        # 임시 파일 경로 반환
        return {"status": "success", "temp_file_path": temp_file_path}
    except Exception as e:
        return {"status": "failed", "message": str(e)}