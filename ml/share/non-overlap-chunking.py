import os
from pathlib import Path
from copy import deepcopy
from math import ceil
from tqdm import tqdm
from miditoolkit import MidiFile

base_path = '/home/level2-3-nlp-finalproject-nlp-07/ml/'
data_path = base_path + 'data/full/'
midi_data_path = "lakh_clean_midi"

full_path = data_path + midi_data_path

midi_paths = list(Path(full_path).rglob('*.mid'))

MAX_NB_BAR = 4
# MIN_NB_NOTES = 20

output_path = base_path + f'data/chunks{MAX_NB_BAR}/'

for midi_path in tqdm(midi_paths):
    try:
        midi = MidiFile(midi_path)
    except Exception as e:
        print("Skipping", midi_path, "because of the following error:", e)
        continue
    
    if not os.path.exists(output_path + f'{midi_data_path}/{midi_path.stem}'):
        os.makedirs(output_path + f'{midi_data_path}/{midi_path.stem}')
    
    # 1박 * 4박자 * MAX_NB_BAR마디
    ticks_per_cut = midi.ticks_per_beat * 4 * MAX_NB_BAR
    nb_cut = ceil(midi.max_tick / ticks_per_cut)
    if nb_cut < 2:
        # 2마디도 안 나오면 그냥 원본 그대로 저장
        midi.dump(output_path + f'{midi_data_path}/{midi_path.stem}/1.mid')
        continue
    
    # MAX_NB_BAR마디 단위로 청킹    
    midi_cuts = [deepcopy(midi) for _ in range(nb_cut)]
    
    # Chunking
    for j, track in enumerate(midi.instruments):
        track.notes = sorted(track.notes, key=lambda x: x.start)
        for midi_short in midi_cuts:
            midi_short.instruments[j].notes = []
        for note in track.notes:
            cut_idx = note.start // ticks_per_cut
            note_copy = deepcopy(note)
            note_copy.start -= cut_idx * ticks_per_cut
            note_copy.end -= cut_idx * ticks_per_cut
            midi_cuts[cut_idx].instruments[j].notes.append(note_copy)            
            
    # Saving
    for j, midi_short in enumerate(midi_cuts):
        # if sum(len(track.notes) for track in midi_short.instruments) < MIN_NB_NOTES:
        #     print("Skipping", midi_path, "because it's too short")
        #     continue
        midi_short.dump(output_path + f'{midi_data_path}/{midi_path.stem}/{j+1}.mid')