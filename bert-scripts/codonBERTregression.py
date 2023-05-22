#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
import pickle
import os
import gc
import helpers

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig, DistilBertPreTrainedModel, DistilBertModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

PATH='/lustre/isaac/proj/UTK0196/codon-expression-data/fullTableForTrainning/'
RUN=1
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def compute_metrics(epred):
    # Computes metrics from specialized output from huggingface
    logits = epred[0]
    labels = epred[1]

    metrics = {}

    metrics['mse'] = mean_squared_error(labels, logits)
    metrics['mae'] = mean_absolute_error(labels, logits)
    metrics['r2'] = r2_score(labels, logits)
    #metrics['single_squared_errors'] = ((logits - labels).flatten()**2).tolist()

    return metrics

class BertRegresser(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = DistilBertModel(config)
        #The output layer that takes the [CLS] representation and gives an output
        self.cls_layer1 = torch.nn.Linear(config.hidden_size,128)
        self.relu1 = torch.nn.ReLU()
        self.ff1 = torch.nn.Linear(128,128)
        self.tanh1 = torch.nn.Tanh()
        self.ff2 = torch.nn.Linear(128,1)

    def forward(self, input_ids, attention_mask, labels):
        #Feed the input to Bert model to obtain contextualized representations
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #Obtain the representations of [CLS] heads
        logits = outputs.last_hidden_state[:,0,:]
        output = self.cls_layer1(logits)
        output = self.relu1(output)
        output = self.ff1(output)
        output = self.tanh1(output)
        output = self.ff2(output)
        return output


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        #logits = outputs.get("logits")

        #print(labels, outputs)

        # compute custom loss
        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(labels, outputs)
        return (loss, outputs) if return_outputs else loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Tells the model we need to use the GPU

print('Reading data...')
filelist = os.listdir(PATH) 
df_list = [pd.read_csv(PATH+file) for file in filelist]
df = pd.concat(df_list)

df = helpers.add_codons_to_df(df, 'Sequence')
labels = df['median_exp']

#labels = labels.type(torch.LongTensor)

classification_df = pd.DataFrame({'text' : df['codons_cleaned'], 'label' : labels})
#MAX = int(max([(len(elem) / 3) for elem in df['codons_cleaned']])) #get max sequence length for padding
#MED = int(np.median([(len(elem) / 3) for elem in df['codons_cleaned']])) #get median sequence length for padding
#print(MED)
#trunc_len = int((MAX + MED) / 2) #set truncation somewhere between max and median
trunc_len = 1064

df_train, df_test = train_test_split(classification_df, test_size=0.2, random_state=1234)
df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=1234)

ds_train = Dataset.from_pandas(df_train)
ds_val = Dataset.from_pandas(df_val)
ds_test = Dataset.from_pandas(df_test)

del classification_df
del df_train
del df_test
del df_val

print('Tokenizing...')
config = AutoConfig.from_pretrained('distilbert-base-uncased', max_position_embeddings=trunc_len)
tokenizer = AutoTokenizer.from_pretrained('./tokenizers/codonBERT2', model_max_length=trunc_len, padding_side='left', truncation_side='right')


tokenized_ds_train = ds_train.map(lambda d : tokenizer(d['text'], truncation=True, padding=True), batched=True)
tokenized_ds_val = ds_val.map(lambda d : tokenizer(d['text'], truncation=True, padding=True), batched=True)
tokenized_ds_test = ds_test.map(lambda d : tokenizer(d['text'], truncation=True, padding=True), batched=True)
del ds_train
del ds_val
del ds_test
gc.collect()

torch.cuda.empty_cache()

print('Building Model...')
#model = AutoModelForSequenceClassification.from_pretrained('./models/codonBERT-binary_1/checkpoint-99500', num_labels=1, ignore_mismatched_sizes=True)
#model = AutoModelForSequenceClassification.from_config(config) #randomly initialize it

## Creating the model from the desired transformer model
model = BertRegresser.from_pretrained('./models/codonBERT-binary_reg/checkpoint-99500', config=config)
#optimizer = torch.optim.Adam(params=model.parameters())

training_args = TrainingArguments(
    output_dir='./models/codonBERT-binary_reg_1'.format(RUN),
    learning_rate=2e-6,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds_train,
    eval_dataset=tokenized_ds_val,
    tokenizer=tokenizer,
)


print('Training...')
trainer.train()
trainer.evaluate()
out = trainer.predict(test_dataset=tokenized_ds_test)

scores = compute_metrics(out)
with open('./results/codonBERT-binary_reg_1.txt'.format(RUN),'w') as data: 
      data.write(str(scores))

