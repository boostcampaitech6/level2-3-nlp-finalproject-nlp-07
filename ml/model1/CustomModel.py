from transformers import (BertPreTrainedModel, RobertaPreTrainedModel, RobertaModel)
from torch import nn
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import SequenceClassifierOutput
import pickle
import torch
from tqdm import tqdm



class customRobertaForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels1 = None, num_labels2 = None, num_labels3 = None ):
        super().__init__(config)
        self.num_labels1 = config.num_labels1
        self.num_labels2 = config.num_labels2
        self.num_labels3 = config.num_labels3
        
        self.config = config
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense3 = nn.Linear(config.hidden_size, config.hidden_size)
        
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
        
        output1 = self.dropout(pooled_output)
        output1 = self.dense1(pooled_output)
        output1 = torch.tanh(output1)
        output1 = self.dropout(output1)

        output2 = self.dropout(pooled_output)
        output2 = self.dense2(pooled_output)
        output2 = torch.tanh(output2)
        output2 = self.dropout(output2)

        output3 = self.dropout(pooled_output)
        output3 = self.dense3(pooled_output)
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
    


def data_labels(label_data_path ='./labels.pkl'):
    with open(label_data_path,'rb') as f:
        labels = pickle.load(f)
    
    return labels['emotion_labels'], labels['tempo_labels'], labels['genre_labels']
    

class frontModelDataset:
    def __init__(self, data, tokenizer, label_data_path ='./labels.pkl'):

        emotion_labels, tempo_labels, genre_labels= data_labels(label_data_path)
        
        id2label_emotion = {k:l for k, l in enumerate(emotion_labels)}
        label2id_emotion = {l:k for k, l in enumerate(emotion_labels)}
        id2label_tempo = {k:l for k, l in enumerate(tempo_labels)}
        label2id_tempo = {l:k for k, l in enumerate(tempo_labels)}
        id2label_genre = {k:l for k, l in enumerate(genre_labels)}
        label2id_genre = {l:k for k, l in enumerate(genre_labels)}

        self.tokenizer = tokenizer
        self.dataset = []
        datas = []
        self.labels1 = []
        self.labels2 = []
        self.labels3 = []
        for idx, df in tqdm(data.iterrows()):
            label1 = [0. for _ in range(len(id2label_emotion))]
            label2 = [0. for _ in range(len(id2label_tempo))]
            label3 = [0. for _ in range(len(id2label_genre))]
            datas.append(df.caption)
            label1[label2id_emotion[df.emotion]] = 1.
            label2[label2id_tempo[df['tempo(category)']]] = 1.
            label3[label2id_genre[df['genre']]] = 1.
            self.labels1.append(label1)
            self.labels2.append(label2)
            self.labels3.append(label3)
        
        self.dataset =  tokenizer(datas,padding=True, truncation=True,max_length=512 ,return_tensors="pt").to('cuda')
        self.labels1= torch.tensor(self.labels1)
        self.labels2= torch.tensor(self.labels2)
        self.labels3= torch.tensor(self.labels3)

    def __len__(self):
        return len(self.labels1)
    
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.dataset.items()}
        item['labels1'] = self.labels1[idx].clone().detach()
        item['labels2'] = self.labels2[idx].clone().detach()
        item['labels3'] = self.labels3[idx].clone().detach()
        return item