import transformers
from transformers import AutoModelForCausalLM
# from IPython.display import Audio
from anticipation import ops
from anticipation.ops import get_instruments
from anticipation.sample import generate
# from anticipation.tokenize import extract_instruments
from anticipation.convert import events_to_midi,midi_to_events
# from anticipation.visuals import visualize
from anticipation.config import *
from anticipation.vocab import *
# from anticipation.ops import get_instruments
from symusic import Score
from symusic.core import TempoTick
from pathlib import Path
import os
from settings import TEMP_DIR, ANTICIPATION_MODEL_NAME
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def initialize_anticipation_model():
    model = AutoModelForCausalLM.from_pretrained(ANTICIPATION_MODEL_NAME).to(DEVICE)
    return model

def get_instruments_list(proposal):
    return list(get_instruments(proposal).keys())

# 미디파일 정보 저장 함수
def extract_midi_info(MIDI_FILE):   
    score = Score(Path(MIDI_FILE))
    qpm_tmp = score.tempos[0].qpm   # bpm
    mspq = score.tempos[0].mspq     # mspq
    return mspq, qpm_tmp

# 120bpm으로 바꿔주는 함수: bpm을 무조건 120으로 바꿔 처리하므로, 120bpm으로 변환 
def pre_processing(MIDI_FILE):  
    score = Score(Path(MIDI_FILE))
    new_tempo_tick = TempoTick(time=0, qpm=120, mspq=50000)
    score.tempos[0] = new_tempo_tick
    score.dump_midi('./anticipation-temp.mid')    
    evnets = midi_to_events('./anticipation-temp.mid')
    os.remove('./anticipation-temp.mid')
    
    return evnets

def extend_4bar_to_8bar(model, midi_path):
    length = 60 / 120 * 4 * 4
    mspq, qpm = extract_midi_info(midi_path)
    events = pre_processing(midi_path)
    history = events.copy()
    n = length
    top_p = 0.9
    proposal = generate(model, start_time=length, end_time=length+n, inputs=history, top_p=top_p)
    inst_num = get_instruments_list(history)
    proposal = ops.delete(proposal, lambda token: (token[2]-NOTE_OFFSET) // 2**7 not in inst_num)
    proposal_midi = events_to_midi(proposal)

    temp_file_path = os.path.join(TEMP_DIR, "anticipation-extended.mid")
    proposal_midi.save(temp_file_path)
    extended_midi = Score(temp_file_path)
    
    # new_tempo_tick = TempoTick(time=0, qpm=qpm, mspq=mspq)
    # extended_midi.tempos.append(new_tempo_tick)

    return extended_midi

def infill_at(model, midi_path, bar_idx, num_of_bars = 8):
    if bar_idx > num_of_bars or bar_idx < 2:
        print("bar_idx must be in range 2 ~ 8")
        # raise ValueError("bar_idx must be in range 2 ~ 8")
    
    length = 60 / 120 * 4 * 4
    length_per_bar = (length/num_of_bars)*(num_of_bars/4) # 사실 항상 2임
    mspq, qpm = extract_midi_info(midi_path)
    events = pre_processing(midi_path)
    history = events.copy()

    # e.g. 0 ~ 4마디                    
    segment = ops.clip(history , 0, length_per_bar*(bar_idx-1) , clip_duration=False)
    # e.g. 5 ~ 8마디
    anticipated = [CONTROL_OFFSET + tok for tok in ops.clip(history , length_per_bar*bar_idx, length*num_of_bars/4, clip_duration=False)] 

    # 5번째 마디 생성 & 맘에 안들 경우, 여기서부터 re-run
    inpainted = generate(model, length_per_bar*(bar_idx-1), length_per_bar*bar_idx, inputs=segment, controls=anticipated, top_p=.95)
    proposal = ops.combine(inpainted, anticipated)
    inst_num = get_instruments_list(history)
    proposal = ops.delete(proposal, lambda token: (token[2]-NOTE_OFFSET) // 2**7 not in inst_num)
    proposal_midi = events_to_midi(proposal)

    temp_file_path = os.path.join(TEMP_DIR, "anticipation-infilled.mid")
    proposal_midi.save(temp_file_path)
    infilled_midi = Score(temp_file_path)
    # new_tempo_tick = TempoTick(time=0, qpm=qpm, mspq=mspq)
    # infilled_midi.tempos.append(new_tempo_tick)
    
    return infilled_midi