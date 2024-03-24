import pandas as pd
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoConfig, EarlyStoppingCallback
import evaluate
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report
from CustomModel import customRobertaForSequenceClassification, frontModelDataset
from CustomTrainer import CustomTrainer, custom_compute_metrics
from transformers.configuration_utils import PretrainedConfig
import wandb
import random

def set_seed(seed:int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    set_seed(42)

    
    data = pd.read_csv('./origin_llama2_when.csv')

    BASE_MODEL = 'SamLowe/roberta-base-go_emotions'

    labels = {'emotion_labels' :data.emotion.unique(), 'tempo_labels' : data['tempo(category)'].unique(),
              'genre_labels' : data['genre'].unique() }

    with open('labels.pkl','wb') as f:
        pickle.dump(labels, f)




    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    config = AutoConfig.from_pretrained(BASE_MODEL)
    
    config.num_labels1 = len(labels['emotion_labels'])
    config.num_labels2 = len(labels['tempo_labels'])
    config.num_labels3 = len(labels['genre_labels'])

    model = customRobertaForSequenceClassification.from_pretrained(BASE_MODEL, config= config).to(device)

    
    ## Data split 

    data_valid_index = data.groupby(['emotion','genre','tempo(category)']).sample(frac=0.1, random_state=42).index
    valid_data = data.iloc[data_valid_index]
    train_data = data.drop(list(data_valid_index)).sample(frac=1, random_state=42)

    dataset_train = frontModelDataset(train_data, tokenizer =tokenizer)
    dataset_valid = frontModelDataset(valid_data, tokenizer =tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    
    training_args = TrainingArguments(

        output_dir="my_awesome_model",
        save_steps=300,
        eval_steps = 300, 
        warmup_steps=500,
        logging_steps=100,
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=8,
        weight_decay=0.01,
        evaluation_strategy='steps',
        load_best_model_at_end = True,
        save_total_limit = 2,
        report_to="wandb",
        metric_for_best_model='fi_total',
        # run_name=BASE_MODEL, 
        )

    trainer = CustomTrainer(

        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics = custom_compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
        )


if __name__ == '__main__':
    main()

