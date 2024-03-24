import transformers
from transformers import AutoModelForCausalLM
from IPython.display import Audio
from anticipation import ops
from anticipation.sample import generate
from anticipation.tokenize import extract_instruments
from anticipation.convert import events_to_midi,midi_to_events
from anticipation.visuals import visualize
from anticipation.config import *
from anticipation.vocab import *
from anticipation.ops import get_instruments

SMALL_MODEL = 'stanford-crfm/music-small-800k'   # slower inference, better sample quality
model = AutoModelForCausalLM.from_pretrained(SMALL_MODEL).cuda()

def extract_midi_info(MIDI_FILE):
    score = Score(Path(MIDI_FILE))
    qpm = score.tempos[0].qpm
    sec = 60 / qpm * 4 * 4
    length = sec         
    mspq = score.tempos[0].mspq
    return sec, length, mspq


sec, length, mspq = extract_midi_info(MIDI_FILE)

start_position = 0 

events = midi_to_events(MIDI_FILE)
history = events.copy()

n = sec
top_p = 0.9
iter = 0     # iter : 현재 마디 수 - 1

proposal = generate(model, start_time=length, end_time=length+n, inputs=history, top_p=top_p)

mid = events_to_midi(proposal)