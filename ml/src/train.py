import random
import numpy as np
import torch

from transformers import AutoConfig, GPT2LMHeadModel, set_seed, EarlyStoppingCallback, TrainingArguments
from tokenizer import get_custom_tokenizer
from miditok.pytorch_data import DataCollator
from load_data import load_midi_paths, CodeplayDataset, chunk_midi, overlap_chunk_midi
from trainer import CodeplayTrainer
from utils import split_train_valid
from datetime import datetime

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

# MAX_SEQ_LEN, BATCH_SIZE = 2048, 8 # 8마디 용
MAX_SEQ_LEN, BATCH_SIZE = 1024, 16
# MAX_SEQ_LEN, BATCH_SIZE = 512, 32

def main():
    tokenizer = get_custom_tokenizer()
    
    midi_paths = ['../data/full/jazz-midi-clean']
    print(midi_paths)
    midi_paths = load_midi_paths(midi_paths)
    print('num of midi files:', len(midi_paths))
    train_midi_paths, valid_midi_paths = split_train_valid(midi_paths, valid_ratio=0.05, shuffle=True, seed=SEED)
    print('num of train midi files:', len(train_midi_paths), 'num of valid midi files:', len(valid_midi_paths))
    
    # midi paths to midi chunks
    # train_midi_chunks = chunk_midi(train_midi_paths)
    # valid_midi_chunks = chunk_midi(valid_midi_paths)
    
    train_midi_chunks = overlap_chunk_midi(train_midi_paths, chunk_bar_num=4, overlap=2)
    valid_midi_chunks = overlap_chunk_midi(valid_midi_paths, chunk_bar_num=4, overlap=2)
    
    # midi chunks to midi tokens
    train_dataset = CodeplayDataset(midis=train_midi_chunks, min_seq_len=50, max_seq_len=MAX_SEQ_LEN-2, tokenizer=tokenizer)
    valid_dataset = CodeplayDataset(midis=valid_midi_chunks, min_seq_len=50, max_seq_len=MAX_SEQ_LEN-2, tokenizer=tokenizer)
    collator = DataCollator(tokenizer["PAD_None"], tokenizer["BOS_None"], tokenizer["EOS_None"], copy_inputs_as_labels=True)
    print('tokenized train_dataset:', len(train_dataset), 'tokenized valid_dataset:', len(valid_dataset))

    #TODO -  context length는 자유롭게 바꿔보며 실험해봐도 좋을 듯 합니다.
    context_length = MAX_SEQ_LEN

    #TODO: Change this based on size of the data
    n_layer=12
    n_head=12
    n_emb=384

    # gpt2 config
    model_config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_positions=context_length,
        n_layer=n_layer,
        n_head=n_head,
        pad_token_id=tokenizer["PAD_None"],
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
        n_embd=n_emb
    )

    #NOTE - nvidia update 필요합니다!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    
    model = GPT2LMHeadModel(model_config)
    model.to(device)
    
    
    # Get the output directory with timestamp.
    # output_path with timestamp
    DATASET_NAME = 'jazz-midi-clean'
    output_path = "../models/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + DATASET_NAME
    steps = 400
    # Commented parameters correspond to the small model
    trainer_config = {
        "output_dir": output_path,
        "num_train_epochs": 30, # 학습 epoch 자유롭게 변경. 저는 30 epoch 걸어놓고 early stopping 했습니다.
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
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