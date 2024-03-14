import torch
from fastapi import APIRouter, Form, File, UploadFile, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from symusic import Score
import os, shutil
import logging
# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from utils.utils import clear_huggingface_cache
from utils.generateModel_util import initialize_generate_model, generate_initial_track, generate_additional_track
from utils.frontModel_util import initialize_front_model, extract_condition
from utils.tokenizer_converter import mmm_to_nnn, nnn_to_mmm
from utils.data_processing import get_instruments_for_generate_model
from settings import TEMP_DIR

# 캐쉬 삭제
clear_huggingface_cache(False)

class TextData(BaseModel):
    prompt: str

router = APIRouter()

# front model initialize
front_model, front_tokenizer = initialize_front_model()

# # generate model initialize
# generate_model, generate_tokenizer = initialize_generate_model()

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

    # condition 추출
    condition = extract_condition(text, front_model, front_tokenizer)
    logging.info("emotion : %s,  tempo : %s,  genre : %s", *condition)
    
    genre_instruments = get_instruments_for_generate_model(condition)
    logging.info(genre_instruments)
    
    # ## generation midi
    # generated_ids = generate_initial_track(generate_model, generate_tokenizer, temperature=0.8)

    # mmm_tokens_ids = nnn_to_mmm(generated_ids[0].tolist(), generate_tokenizer)
    # midi_data = generate_tokenizer.tokens_to_midi(mmm_tokens_ids)

    # file_path = os.path.join(TEMP_DIR, "temp_gen.mid")
    # midi_data.dump_midi(file_path)

    # return FileResponse(file_path, media_type="audio/midi")
    # return StreamingResponse(open(file_path, "rb"), media_type="audio/midi")

@router.post("/upload_midi/")
async def receive_midi(midi_file: UploadFile = File(...), instnum: int = Form(...)):
    logging.info(f"midi_file : {midi_file}")
    logging.info(f"instnum : {instnum}")

    temp_file_path = os.path.join(TEMP_DIR, "temp_receive.mid")
    try:
        # 업로드된 파일을 임시 폴더에 저장
        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(midi_file.file, temp_file)
        midi = Score(temp_file_path)
    except Exception as e:
        return {"status": "failed", "message": str(e)}
    
    # mmm_tokens_ids = generate_tokenizer(midi).ids
    # nnn_tokens_ids = mmm_to_nnn(mmm_tokens_ids, generate_tokenizer)
    # nnn_tokens_ids = torch.tensor([nnn_tokens_ids])
    
    # generated_ids = generate_additional_track(nnn_tokens_ids, generate_model, generate_tokenizer, temperature=0.8)
    # mmm_generated_ids = nnn_to_mmm(generated_ids[0].tolist(), generate_tokenizer)
    # midi_data = generate_tokenizer.tokens_to_midi(mmm_generated_ids)

    # file_path = os.path.join(TEMP_DIR, "temp_additional.mid")
    # midi_data.dump_midi(file_path)
    
    # return FileResponse(file_path, media_type="audio/midi")