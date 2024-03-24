from transformers import Trainer
from torch import nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from transformers.utils import is_sagemaker_mp_enabled, is_apex_available
from sklearn.metrics import f1_score, accuracy_score
import numpy as np


class CustomTrainer(Trainer):
    def __init__(self, group_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.group_weights = group_weights
        
    def compute_loss(self, model, inputs, return_outputs=False ):

        labels_emotion = inputs.pop(f"labels1")
        labels_tempo = inputs.pop(f"labels2")
        labels_genre = inputs.pop(f"labels3")
        
        outputs = model(**inputs)

        logits_emotion = outputs[0][0]
        logits_tempo = outputs[0][1]
        logits_genre = outputs[0][2]

        loss_emotion = torch.nn.functional.binary_cross_entropy_with_logits(logits_emotion, labels_emotion)
        loss_tempo = torch.nn.functional.binary_cross_entropy_with_logits(logits_tempo, labels_tempo)
        loss_genre = torch.nn.functional.binary_cross_entropy_with_logits(logits_genre, labels_genre)
        loss = loss_emotion + loss_tempo + loss_genre

        return (loss, outputs) if return_outputs else loss
    


def get_preds_from_logits(logits):
    ret = np.zeros(logits.shape)
    
    best_class = np.argmax(logits, axis=1)
    ret[list(range(len(ret))), best_class] = 1
    
    return ret
    

def custom_compute_metrics(eval_pred):
    logits, labels = eval_pred
    final_metrics = {}
    
    # Deduce predictions from logits
    predictions_emotion = get_preds_from_logits(logits[0])
    predictions_tempo = get_preds_from_logits(logits[1])
    predictions_genre = get_preds_from_logits(logits[2])
    
    # Get f1 metrics for global scoring. Notice that f1_micro = accuracy
    final_metrics["f1_emotion"] = f1_score(labels[0], predictions_emotion, average="micro")
    
    # Get f1 metrics for causes
    final_metrics["f1_tempo"] = f1_score(labels[1], predictions_tempo, average="micro")
    

    # The global f1_metrics
    final_metrics["f1_genre"] = f1_score(labels[2], predictions_genre, average="micro")

    final_metrics['fi_total'] = (final_metrics["f1_emotion"] + final_metrics["f1_tempo"] + final_metrics["f1_genre"])/3
    
    # Classification report
    return final_metrics
    
