import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import AutoConfig, GPT2LMHeadModel, set_seed, EarlyStoppingCallback, TrainingArguments
from tokenizer import get_custom_tokenizer
from miditok.pytorch_data import DataCollator
from load_data import load_midi_paths, CodeplayDataset
from trainer import CodeplayTrainer
from utils import split_train_valid


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

def main():
    tokenizer = get_custom_tokenizer()
    
    midi_paths = ['../data/chunks/']
    midi_paths = load_midi_paths(midi_paths)
    random.shuffle(midi_paths)
    train_midi_paths, valid_midi_paths = split_train_valid(midi_paths)
    
    # midi_paths to midi to tokens
    train_dataset = CodeplayDataset(files_paths=train_midi_paths, min_seq_len=50, max_seq_len=1022, tokenizer=tokenizer)
    valid_dataset = CodeplayDataset(files_paths=valid_midi_paths, min_seq_len=50, max_seq_len=1022, tokenizer=tokenizer)
    collator = DataCollator(tokenizer["PAD_None"], tokenizer["BOS_None"], tokenizer["EOS_None"], copy_inputs_as_labels=True)

    # context length는 자유롭게 바꿔보며 실험해봐도 좋을 듯 합니다.
    context_length = 1024 

    #TODO: Change this based on size of the data
    n_layer=6
    n_head=4
    n_emb=1024

    config = AutoConfig.from_pretrained(
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
    model = GPT2LMHeadModel(config)
    model.to(device)
    
    
    # Get the output directory with timestamp.
    output_path = "../models"
    steps = 100
    # Commented parameters correspond to the small model
    config = {"output_dir": output_path,
            "num_train_epochs": 30, # 학습 epoch 자유롭게 변경. 저는 30 epoch 걸어놓고 early stopping 했습니다.
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
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
    
    train_args = TrainingArguments(**config)

    #TODO - DataCollator customize
    trainer = CodeplayTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)] # Early Stopping patience 자유롭게 변경
    )
    
    trainer.train()    

if __name__ == '__main__':
    print('Training model...')
    main()