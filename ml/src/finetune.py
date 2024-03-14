import random
import numpy as np
import pandas as pd
import torch

from transformers import AutoConfig, GPT2LMHeadModel, set_seed, EarlyStoppingCallback, TrainingArguments
from tokenizer import get_custom_tokenizer, get_nnn_tokenizer, get_nnn_meta_tokenizer
from miditok.pytorch_data import DataCollator
from load_data import load_midi_paths, CodeplayDataset, chunk_midi, overlap_chunk_midi, meta_chunk_midi
from trainer import CodeplayTrainer
from utils import split_train_valid
from datetime import datetime

from pathlib import Path

SEED = 2024
deterministic = False

#REVIEW - seed 고정할 것 더 있나요?
random.seed(SEED) # python random seed 고정
np.random.seed(SEED) # numpy random seed 고정
torch.manual_seed(SEED) # torch random seed 고정
torch.cuda.manual_seed_all(SEED) # torch random seed 고정
set_seed(SEED) # transformers random seed 고정
if deterministic: # cudnn random seed 고정 - 고정 시 학습 속도가 느려질 수 있습니다. 
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

args_data = {
    "Jazz": "../data/full/jazz-midi-clean",
    "Lakh": "../data/full/lakh_clean_midi",
}

def main():
    args = {}
    # args["user"] = "devBuzz142"
    # args["title"] = "lakh-NNN"
    
    # MAX_SEQ_LEN, BATCH_SIZE = 2048, 8 # 8마디 용
    # MAX_SEQ_LEN, BATCH_SIZE = 1024, 16
    args["max_seq_len"] = 1024
    args["batch_size"] = 16
    
    args["tokenizer"] = "NNN-vel4"
    if args["tokenizer"] == "NNN-vel4":
        tokenizer = get_nnn_tokenizer(4)
    elif args["tokenizer"] == "NNN-vel8":
        tokenizer = get_nnn_tokenizer(8)
    elif args["tokenizer"] == "MMM":
        tokenizer = get_custom_tokenizer()
    elif args["tokenizer"] == "NNN-meta":
        tokenizer = get_nnn_meta_tokenizer(4)
    
    #NOTE - sampled
    fine_tune_data_path = '../../data/full/lakh_clean_midi_sampled'
    metas = pd.read_csv('../data/full/lakh_clean_midi.csv')
    metas = metas[['emotion', 'tempo(int)', 'genre', 'file_path']]

    midi_paths = [[Path(fine_tune_data_path, row["file_path"]), row["genre"], row["emotion"], row["tempo(int)"]] for i, row in metas.iterrows() if Path(fine_tune_data_path, row['file_path']).exists()]
    print('num of midi files:', len(midi_paths))
    train_midi_paths, valid_midi_paths = split_train_valid(midi_paths, valid_ratio=0.05, shuffle=True, seed=SEED)
    print('num of train midi files:', len(train_midi_paths), 'num of valid midi files:', len(valid_midi_paths))
    
    args["chunks_bar_num"] = 4
    args["overlap"] = 0
    train_midi_chunks = meta_chunk_midi(train_midi_paths, chunk_bar_num=args["chunks_bar_num"], overlap=args["overlap"])
    valid_midi_chunks = meta_chunk_midi(valid_midi_paths, chunk_bar_num=args["chunks_bar_num"], overlap=args["overlap"])
    
    # midi chunks to midi tokens
    train_dataset = CodeplayDataset(midis=train_midi_chunks, min_seq_len=50, max_seq_len=args["max_seq_len"]-2, tokenizer=tokenizer)
    valid_dataset = CodeplayDataset(midis=valid_midi_chunks, min_seq_len=50, max_seq_len=args["max_seq_len"]-2, tokenizer=tokenizer)
    collator = DataCollator(tokenizer["PAD_None"], tokenizer["BOS_None"], tokenizer["EOS_None"], copy_inputs_as_labels=True)
    print('tokenized train_dataset:', len(train_dataset), 'tokenized valid_dataset:', len(valid_dataset))

    model_config = AutoConfig.from_pretrained('../models/nnn-vel4-lakh-checkpoint-56000')

    #NOTE - nvidia update 필요합니다!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    
    model = GPT2LMHeadModel(model_config)
    model.to(device)
    
    
    # Get the output directory with timestamp.
    # output_path with timestamp
    # datetime.now().strftime("%Y%m%d-%H%M%S")  
    output_path = f"../models/fine_tune/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
    steps = 400
    # Commented parameters correspond to the small model
    trainer_config = {
        "output_dir": output_path,
        "num_train_epochs": 40, # 학습 epoch 자유롭게 변경. 저는 30 epoch 걸어놓고 early stopping 했습니다.
        "per_device_train_batch_size": args["batch_size"],
        "per_device_eval_batch_size": args["batch_size"],
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "eval_steps": steps,
        "logging_steps":steps,
        "logging_first_step": True,
        "save_total_limit": 5,
        "save_steps": steps,
        "lr_scheduler_type": "cosine",
        "learning_rate":5e-4,
        "warmup_ratio": 0.01,
        "weight_decay": 0.01,
        "seed": SEED,
        "load_best_model_at_end": True,
        # "metric_for_best_model": "eval_loss" # best model 기준 바꾸고 싶을 경우 이 부분 변경 (default가 eval_loss임)
        #   "report_to": "wandb"
    }
    
    train_args = TrainingArguments(**trainer_config)

    #TODO - DataCollator customize
    trainer = CodeplayTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=8)] # Early Stopping patience 자유롭게 변경
    )
    
    trainer.train()    

if __name__ == '__main__':
    print('Training model...')
    main()