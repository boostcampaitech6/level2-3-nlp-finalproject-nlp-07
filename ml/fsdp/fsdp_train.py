# 1. Import necessary packages
import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    CPUOffload,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import time
import tqdm
from datetime import datetime

"""
추가
"""
from tokenizer import get_custom_tokenizer, get_nnn_tokenizer
# from transformers import AutoConfig, GPT2LMHeadModel
from transformers import GPT2LMHeadModel, GPT2Config
from utils import load_dataset
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME
from load_data import CodeplayDataset, overlap_chunk_midi
from miditok.pytorch_data import DataCollator
import wandb
import shutil
    
# 2. Distributed training setup
def setup(args):
    # initialize the process group
    ip = args.master_ip
    os.environ["MASTER_ADDR"] = ip
    dist.init_process_group("nccl")    
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()
    
class MidiModel(GPT2LMHeadModel): 
    def __init__(self, tokenizer, context_length, n_layer, n_head, n_emb):
        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=context_length,
            n_layer=n_layer,
            n_head=n_head,
            pad_token_id=tokenizer["PAD_None"],
            bos_token_id=tokenizer["BOS_None"],
            eos_token_id=tokenizer["EOS_None"],
            n_embd=n_emb,
            # output_hidden_states=True
        )
        super().__init__(config)

def setup_tokenizer(num_velocities, tokenizer_args):
    if tokenizer_args == "mmm" :
        tokenizer = get_custom_tokenizer(num_velocities)
    elif tokenizer_args == "nnn" : 
        tokenizer = get_nnn_tokenizer(num_velocities)
    return tokenizer

def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run

def format_metrics_to_gb(item):
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    g_gigabyte = 2**10
    metric_num = item / g_gigabyte
    metric_num = round(metric_num, ndigits=4)
    return metric_num

def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(2).to(local_rank)

    if sampler:
        sampler.set_epoch(epoch)
    if rank==0:
        inner_pbar = tqdm.tqdm(
            range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
        )
    for batch in train_loader:
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)
        optimizer.zero_grad()
        output = model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],labels=batch["labels"] )
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        fsdp_loss[0] += loss.item()
        fsdp_loss[1] += len(batch)
        if rank==0:
            inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    train_accuracy = fsdp_loss[0] / fsdp_loss[1]


    if rank == 0:
        inner_pbar.close()
        print(
                f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}"
            )
    return train_accuracy

# 5. Define a validation function
def validation(model, rank, world_size, val_loader):
    model.eval()
    correct = 0
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(3).to(local_rank)
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(val_loader)), colour="green", desc="Validation Epoch"
        )
    with torch.no_grad():
        for batch in val_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            output = model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],labels=batch["labels"])
            fsdp_loss[0] += output["loss"].item()  # sum up batch loss
            fsdp_loss[1] += len(batch)

            if rank==0:
                inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    val_loss = fsdp_loss[0] / fsdp_loss[1]
    if rank == 0:
        inner_pbar.close()
        print(f"Validation Loss: {val_loss:.4f}")
    return val_loss

def delete_old_models(models_dir, max_folder_to_keep=5):
    # models 내의 모든 폴더
    all_folders = [os.path.join(models_dir, folder) for folder in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, folder))]
    # 폴더 날짜 별 정렬
    sorted_folders = sorted(all_folders, key=lambda x: os.path.getmtime(x))
    
    if len(sorted_folders) >= max_folder_to_keep :
        # 오래된 폴더 삭제(=models 중 loss 큰 폴더 삭제)
        folders_to_delete = sorted_folders[:-(max_folder_to_keep-len(sorted_folders))]
        for folder in folders_to_delete:
            print(f"Deleting folder: {folder}")
            shutil.rmtree(folder)

# 6.  Define a distributed train function that wraps the model in FSDP
def fsdp_main(args):
    wandb.init(project="fsdp_train", config=args)
    setup(args)
    """
    custom model class 
    """
    context_length = args.context_length
    n_layer=args.n_layer
    n_head=args.n_head
    n_emb=args.n_emb
    
    tokenizer = setup_tokenizer(args.num_velocities, args.tokenizer)
    model = MidiModel(tokenizer, context_length, n_layer, n_head, n_emb)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters count: {total_params}")

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = args.rank
    print("rank: ", rank)
    world_size = args.world_size
    print("world_size: ", world_size)

    
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, '../data')
    dataset = os.path.join(DATA_DIR,args.dataset)
    train_midi_paths, valid_midi_paths = load_dataset(dataset)

    
    chunks_bar_num = args.chunks_bar_num
    overlap = args.overlap
    train_data = overlap_chunk_midi(train_midi_paths, chunk_bar_num=chunks_bar_num, overlap=overlap)
    valid_data = overlap_chunk_midi(valid_midi_paths, chunks_bar_num, overlap=overlap)
    
    
    print("Size of train dataset: ", len(train_data))
    print("Size of Validation dataset: ", len(valid_data))
    print(train_data[:3])
    print(valid_data[:3])

    dataset_train = CodeplayDataset(
        midis=train_data,
        min_seq_len=50,
        # max_seq_len=1022,
        max_seq_len=args.context_length - 2,
        tokenizer=tokenizer,
    )
    dataset_valid = CodeplayDataset(
        midis=valid_data,
        min_seq_len=50,
        # max_seq_len=1022,
        max_seq_len=args.context_length - 2,
        tokenizer=tokenizer,
    )

    collator = DataCollator(
        tokenizer["PAD_None"], tokenizer["BOS_None"], tokenizer["EOS_None"], copy_inputs_as_labels=True
    )

    sampler1 = DistributedSampler(dataset_train, rank=rank, num_replicas=world_size, shuffle=False)
    sampler2 = DistributedSampler(dataset_valid, rank=rank, num_replicas=world_size)

    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1, 'collate_fn':collator}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2, 'collate_fn':collator}
    cuda_kwargs = {'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset_train,**train_kwargs)
    val_loader = torch.utils.data.DataLoader(dataset_valid, **test_kwargs)

    sharding_strategy: ShardingStrategy = ShardingStrategy.SHARD_GRAD_OP #for Zero2 and FULL_SHARD for Zero3
    torch.cuda.set_device(local_rank)
    
    bfSixteen = MixedPrecision(
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
    )

    # model is on CPU before input to FSDP
    model = FSDP(model,
        mixed_precision=bfSixteen,
        auto_wrap_policy=size_based_auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        cpu_offload=CPUOffload(offload_params=True),
        #sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device())

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    best_val_loss = float("inf")
    curr_val_loss = float("inf")
    file_save_name = "GPT-model"

    if rank == 0:
        time_of_run = get_date_of_run()
        dur = []
        train_acc_tracking = []
        val_acc_tracking = []
        training_start_time = time.time()

    if rank == 0 and args.track_memory:
        mem_alloc_tracker = []
        mem_reserved_tracker = []

    early_stopping_counter = 0
    patience = args.patience
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_accuracy = train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        if args.run_validation:
            curr_val_loss = validation(model, rank, world_size, val_loader)
        scheduler.step()

        if rank == 0:
            wandb.log({'train_loss': train_accuracy.item()})        # train_loss logging
            wandb.log({'validation_loss': curr_val_loss.item()})    # valid_loss logging
            curr_lr = optimizer.param_groups[0]['lr']
            wandb.log({'learning_rate': curr_lr})                   # learning rate logging

            print(f"--> epoch {epoch} completed...entering save and stats zone")

            dur.append(time.time() - t0)
            train_acc_tracking.append(train_accuracy.item())

            if args.run_validation:
                val_acc_tracking.append(curr_val_loss.item())

            if args.track_memory:
                mem_alloc_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_allocated())
                )
                mem_reserved_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_reserved())
                )
            print(f"completed save and stats zone...")

        if args.save_model and curr_val_loss < best_val_loss:

            # save
            if rank == 0:
                print(f"--> entering save model state")

            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = model.state_dict()
                model_to_save = model.module if hasattr(model, 'module') else model

            if rank == 0:
                print(f"--> saving model ...")
                currEpoch = (
                    "-" + str(epoch) + "-" + str(round(curr_val_loss.item(), 4))
                )
                print(f"--> attempting to save model prefix {currEpoch}")
                save_folder = file_save_name + "-" + time_of_run + "-" + currEpoch
                
                OUTPUT_DIR = os.path.join(BASE_DIR, f'../models/{save_folder}')
                if not os.path.exists(OUTPUT_DIR):
                    os.makedirs(OUTPUT_DIR)
                # save_name = "pytorch_model.bin"
                print(f"--> saving as model name {WEIGHTS_NAME} in {OUTPUT_DIR} ")
                
                max_folder_to_keep=5
                delete_old_models(os.path.join(BASE_DIR, '../models'), max_folder_to_keep)
                torch.save(cpu_state, os.path.join(OUTPUT_DIR, WEIGHTS_NAME))     
                model_to_save.config.to_json_file(os.path.join(OUTPUT_DIR, CONFIG_NAME))
                tokenizer.save_pretrained(OUTPUT_DIR)

        if curr_val_loss < best_val_loss:
            if args.earlystopping : 
                early_stopping_counter = 0
            best_val_loss = curr_val_loss
            if rank==0:
                print(f"-->>>> New Val Loss Record: {best_val_loss}")
        else :
            if args.earlystopping : 
                early_stopping_counter += 1 
                print(f"EarlyStopping Counter: {early_stopping_counter} out of {patience}")
                if early_stopping_counter >= patience:
                    print(f"Validation loss hasn't improved for {patience} epochs. Early stopping...")
                    break  # Break out of the training loop

    dist.barrier()
    cleanup()
    
    wandb.finish()
    
# 7. Parse the arguments and set the main function
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch T5 FSDP Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=.002, metavar='LR',
                        help='learning rate (default: .002)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--track_memory', action='store_false', default=False,
                        help='track the gpu memory')
    parser.add_argument('--run_validation', action='store_false', default=True,
                        help='running the validation')
    parser.add_argument('--save-model', action='store_false', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--master_ip', type=str)
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--context_length', type=int, default=1024)
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_emb', type=int, default=512)
    parser.add_argument('--dataset', type=str, default = "kpop-test",
                        help="Place dataset in the \'dataset folder\'")
    parser.add_argument('--chunks_bar_num', type=int, default = 4)
    parser.add_argument('--overlap', type=int, default = 0,
                        help="0(default) means not working")
    parser.add_argument('--num_velocities', type=int, default = 4)
    parser.add_argument('--earlystopping', default = True)
    parser.add_argument('--patience', type=int, default = 3)
    parser.add_argument('--tokenizer', type=str, default="nnn")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    fsdp_main(args)
    
# 8. command
"""
[Master Node 명령어]

NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=eth0,eth1 \
torchrun --nproc_per_node=1 --nnodes=3 --node_rank=0 \
--rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=10.0.1.6:12345 \
fsdp_train.py \
--master_ip 10.0.1.6 --world_size 3 --rank 0 \
--context_length 1024 --n_layer 8 --n_head 8 --n_emb 512


[Worker Node 1 명령어]

NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=eth0 \
torchrun --nproc_per_node=1 --nnodes=3 --node_rank=1 \
--rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=10.0.1.6:12345 \
fsdp_train.py \
--master_ip 10.0.1.6 --world_size 3 --rank 1 \
--context_length 1024 --n_layer 8 --n_head 8 --n_emb 512


[Worker Node 2 명령어]

NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=eth1 \
torchrun --nproc_per_node=1 --nnodes=3 --node_rank=2 \
--rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=10.0.1.6:12345 \
fsdp_train.py --master_ip 10.0.1.6 --world_size 3 --rank 2 \
--context_length 1024 --n_layer 8 --n_head 8 --n_emb 512
"""