import random
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import os
from pathlib import Path

from transformers import AutoConfig, GPT2LMHeadModel, set_seed, EarlyStoppingCallback, TrainingArguments
from tokenizer import get_custom_tokenizer, get_nnn_tokenizer, get_nnn_meta_tokenizer
from miditok.pytorch_data import DataCollator
from load_data import load_midi_paths, CodeplayDataset, chunk_midi
from trainer import CodeplayTrainer
from utils import split_train_valid
from arguments import set_arguments

def set_random_seed(seed=2024, deterministic=False):
    random.seed(seed) # python random seed 고정
    np.random.seed(seed) # numpy random seed 고정
    torch.manual_seed(seed) # torch random seed 고정
    torch.cuda.manual_seed_all(seed) # torch random seed 고정
    set_seed(seed) # transformers random seed 고정
    if deterministic: # cudnn random seed 고정 - 고정 시 학습 속도가 느려질 수 있습니다. 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

#NOTE - 데이터셋을 추가해주세요
datasets = {
    "lakh": "/lakh-clean",
    "jazz": "/jazz-clean",
    "kpop": "/kpop",
    "ALL": "/all",
}

def main(args):
    set_random_seed(args.seed, args.deterministic)
    
    # set tokenizer
    if args.tokenizer == "NNN4":
        tokenizer = get_nnn_tokenizer(4)
    elif args.tokenizer == "MMM":
        tokenizer = get_custom_tokenizer()
    elif args.tokenizer == "NNN-meta":
        tokenizer = get_nnn_meta_tokenizer(4)
        args.use_meta = True
    else:
        raise ValueError("Invalid tokenizer: ", args.tokenizer)
    
    #NOTE - nvidia update 필요합니다!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    #TODO -  context length는 자유롭게 바꿔보며 실험해봐도 좋을 듯 합니다.
    context_length = args.max_seq_len

    # gpt2 config
    model_name = args.model
    model_config = AutoConfig.from_pretrained(
        model_name,
        vocab_size=len(tokenizer),
        n_positions=context_length,
        n_layer=args.n_layer,
        n_head=args.n_head,
        pad_token_id=tokenizer["PAD_None"],
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
        n_embd=args.n_emb
    )
    
    model = GPT2LMHeadModel(model_config)
    model.to(device)
    
    
    # set data path
    if args.dataset not in datasets:
        raise ValueError("Invalid dataset: ", args.dataset)
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, "../data")
    print('dataset dir: ', DATA_DIR + datasets[args.dataset])
    
    # load midi paths
    if args.use_meta:
        metas = pd.read_csv(DATA_DIR + datasets[args.dataset] + f"/meta/{args.meta_file_name}.csv")
        metas = metas[['emotion', 'tempo(category)', 'genre', 'file_path']]
        midi_paths = [
            [Path(DATA_DIR+f'{datasets[args.dataset]}/midi/{row["file_path"]}'), row["genre"], row["emotion"], row["tempo(category)"]]
            for _, row in metas.iterrows()
            if Path(DATA_DIR + datasets[args.dataset] + f'/midi/{row["file_path"]}').exists()
        ]
    else:
        midi_paths = DATA_DIR + datasets[args.dataset]
        midi_paths = load_midi_paths(midi_paths)    
    
    
    print('num of midi files:', len(midi_paths))
    train_midi_paths, valid_midi_paths = split_train_valid(midi_paths, valid_ratio=0.05, shuffle=True, seed=args.seed)
    print('num of train midi files:', len(train_midi_paths), 'num of valid midi files:', len(valid_midi_paths))
    
    # midi to midi chunks
    train_midi_chunks = chunk_midi(train_midi_paths, chunk_bar_num=args.chunk_bar_num, overlap=args.overlap, use_meta=args.use_meta)
    valid_midi_chunks = chunk_midi(valid_midi_paths, chunk_bar_num=args.chunk_bar_num, overlap=args.overlap, use_meta=args.use_meta)
    
    # midi chunks to midi tokens
    train_dataset = CodeplayDataset(midis=train_midi_chunks, min_seq_len=50, max_seq_len=args.max_seq_len-2, tokenizer=tokenizer, use_meta=args.use_meta)
    valid_dataset = CodeplayDataset(midis=valid_midi_chunks, min_seq_len=50, max_seq_len=args.max_seq_len-2, tokenizer=tokenizer, use_meta=args.use_meta)
    collator = DataCollator(tokenizer["PAD_None"], tokenizer["BOS_None"], tokenizer["EOS_None"], copy_inputs_as_labels=True)
    print('tokenized train_dataset:', len(train_dataset), 'tokenized valid_dataset:', len(valid_dataset))

    
    # Get the output directory with timestamp.
    OUTPUT_DIR = os.path.join(BASE_DIR, f"..{args.save_path}")
    save_path = OUTPUT_DIR + f"/{args.model}-{args.dataset}-{args.tokenizer}/{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
    # Commented parameters correspond to the small model
    trainer_config = {
        "output_dir": save_path,
        "num_train_epochs": 40, # 학습 epoch 자유롭게 변경. 저는 30 epoch 걸어놓고 early stopping 했습니다.
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "eval_steps": args.steps,
        "logging_steps":args.steps,
        "logging_first_step": True,
        "save_total_limit": 5,
        "save_steps": args.steps,
        "lr_scheduler_type": "cosine",
        "learning_rate":5e-4,
        "warmup_ratio": 0.01,
        "weight_decay": 0.01,
        "seed": args.seed,
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
    print('===== Training model... =====')
    
    parser = set_arguments('default')
    args = parser.parse_args()
    
    print('==== arguments ====')
    print('model: ', args.model, ', tokenizer: ', args.tokenizer, ', dataset: ', args.dataset)
    print('batch_size: ', args.batch_size, ', max_seq_len: ', args.max_seq_len)
    print('chunk_bar_num: ', args.chunk_bar_num, ', overlap: ', args.overlap)
    print('====================')    

    main(args)