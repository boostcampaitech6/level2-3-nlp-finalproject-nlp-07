import torch
from torch import nn
from transformers import BertPreTrainedModel, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Tuple, Union

from tqdm import tqdm
import pickle

class customRobertaForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels1 = None, num_labels2 = None, num_labels3 = None ):
        super().__init__(config)
        self.num_labels1 = config.num_labels1
        self.num_labels2 = config.num_labels2
        self.num_labels3 = config.num_labels3
        
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier1 = nn.Linear(config.hidden_size, self.num_labels1)
        self.classifier2 = nn.Linear(config.hidden_size, self.num_labels2)
        self.classifier3 = nn.Linear(config.hidden_size, self.num_labels3)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels1: Optional[torch.Tensor] = None,
        labels2: Optional[torch.Tensor] = None,
        labels3: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[0]
        pooled_output = pooled_output[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        output1 = self.dense(pooled_output)
        output1 = torch.tanh(output1)
        output1 = self.dropout(output1)

        output2 = self.dense(pooled_output)
        output2 = torch.tanh(output2)
        output2 = self.dropout(output2)


        output3 = self.dense(pooled_output)
        output3 = torch.tanh(output3)
        output3 = self.dropout(output3)

        logits1 = self.classifier1(output1)
        logits2 = self.classifier2(output2)
        logits3 = self.classifier3(output3)

        loss1 = None
        loss2 = None
        loss3 = None
        
        loss =None
        if loss1 and loss2 and loss3:
            loss = loss1 + loss2 + loss3
        if not return_dict:
            output = (logits1,) + (logits2,) + (logits3,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=(logits1,) + (logits2,) + (logits3,),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
def id2labelData_labels(label_data_path ='./labels.pkl'):
    with open(label_data_path,'rb') as f:
        emotion_labels=pickle.load(f)
        tempo_labels=pickle.load(f)
        genre_labels=pickle.load(f)
    id2label_emotion = {k:l for k, l in enumerate(emotion_labels)}
    id2label_tempo = {k:l for k, l in enumerate(tempo_labels)}
    id2label_genre = {k:l for k, l in enumerate(genre_labels)}
    return id2label_emotion, id2label_tempo, id2label_genre
    
