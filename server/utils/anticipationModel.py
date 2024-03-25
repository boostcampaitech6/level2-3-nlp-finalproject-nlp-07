import transformers
from transformers import AutoModelForCausalLM
# from IPython.display import Audio
# from anticipation import ops
from anticipation.sample import generate
# from anticipation.tokenize import extract_instruments
from anticipation.convert import events_to_midi,midi_to_events
# from anticipation.visuals import visualize
from anticipation.config import *
from anticipation.vocab import *
# from anticipation.ops import get_instruments
from symusic import Score, TempoTick
from pathlib import Path
import os

SMALL_MODEL = 'stanford-crfm/music-small-800k'   # slower inference, better sample quality
model = AutoModelForCausalLM.from_pretrained(SMALL_MODEL).cuda()

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
    temp_midi = score.dump_midi('./anticipation-temp.mid')    
    evnets = midi_to_events(temp_midi)
    os.remove('./anticipation-temp.mid')
    
    return evnets

def extend_4bar_to_8bar(midi_path):
    length = 60 / 120 * 4 * 4
    mspq, qpm = extract_midi_info(midi_path)
    events = pre_processing(midi_path)
    history = events.copy()
    n = length
    top_p = 0.9
    
    proposal = generate(model, start_time=length, end_time=length+n, inputs=history, top_p=top_p)
    proposal_midi = events_to_midi(proposal)
    proposal_midi.save('./anticipation-extended.mid')
    extended_midi = Score(Path('./anticipation-extended.mid'))
    extended_midi.tempos[0].qpm = qpm
    extended_midi.tempos[0].mspq = mspq

    return extended_midi
    