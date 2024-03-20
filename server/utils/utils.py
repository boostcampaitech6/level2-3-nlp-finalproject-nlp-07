import shutil
import os
import mido
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clear_huggingface_cache(delete: bool):
    cache_dir = os.path.expanduser("~/.cache/huggingface/")
    if os.path.exists(cache_dir) and delete:
        try:
            shutil.rmtree(cache_dir)
            logging.info("Hugging Face cache cleared successfully.")
        except Exception as e:
            logging.info(f"Error clearing Hugging Face cache: {e}")
    else:
        logging.info("Hugging Face cache directory not found.")

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)

def extract_tempo(midi_file_path):
    try:
        mid = mido.MidiFile(midi_file_path)
        
        # Tempo 정보가 있는 첫 번째 트랙을 찾음
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    # Tempo 정보를 microseconds per beat 단위로 변환
                    tempo = mido.tempo2bpm(msg.tempo)
                    return tempo
        # 템포 정보가 없는 경우
        print("No tempo information found in MIDI file.")
        return None
    
    except Exception as e:
        print(f"Error: {e}")
        return None
    

def modify_tempo(midi_file_path, new_tempo_bpm):
    try:
        mid = mido.MidiFile(midi_file_path)
        ticks_per_beat = mid.ticks_per_beat
        
        # 새로운 템포 값으로 변환
        new_tempo = mido.bpm2tempo(new_tempo_bpm)
        
        # 템포 메시지 생성
        tempo_msg = mido.MetaMessage('set_tempo', tempo=new_tempo, time=0)
        
        # 첫 번째 트랙의 첫 이벤트로 삽입
        mid.tracks[0].insert(0, tempo_msg)
        
        # 새로운 파일 저장
        mid.save(midi_file_path)
        
        print(f"Tempo modified successfully. New tempo: {new_tempo_bpm} BPM")
    except Exception as e:
        print(f"Error: {e}")