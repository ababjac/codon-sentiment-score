#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
import pickle
import os
import helpers

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split

PATH='/lustre/isaac/proj/UTK0196/codon-expression-data/fullTableForTrainning/'
RUN=0


def compute_metrics(epred):
    # Computes metrics from specialized output from huggingface

    preds = np.argmax(epred[0], axis=1).flatten()
    labels = epred[1].flatten()

    metrics = {}
    #metrics['auprc'] = average_precision_score(labels, preds[:,1])
    metrics['auroc'] = roc_auc_score(labels, preds[:,1], multi_class='ovr', average='micro')
    metrics['precision'] = precision_score(labels, preds[:, 1], average='micro')
    metrics['recall'] = recall_score(labels, preds[:, 1], average='micro')
        
    matrix = multilabel_confusion_matrix(labels, preds[:, 1])
    matrix = [m.flatten() for m in matrix]
    matrix = np.add(matrix)
    metrics['accuracy'] = matrix[0] + matrix[3] / sum(matrix) #categorical accuracy
    return metrics




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Tells the model we need to use the GPU

print('Reading data...')
filelist = os.listdir(PATH) 
df_list = [pd.read_csv(PATH+file) for file in filelist]
df = pd.concat(df_list)

df = helpers.add_codons_to_df(df, 'Sequence')
low, high = df.abundance.quantile([0.33, 0.67])
high_l = np.where(df['abundance'] > high, 2, 0)
low_l = np.where(df['abundance'] > low, 0, 1)
labels = high_l+low_l


classification_df = pd.DataFrame({'text' : df['codons_cleaned'], 'label' : labels})
MAX = int(max([(len(elem) / 3) for elem in df['codons_cleaned']])) #get max sequence length for padding
MED = int(np.median([(len(elem) / 3) for elem in df['codons_cleaned']])) #get median sequence length for padding

trunc_len = int((MAX + MED) / 2) #set truncation somewhere between max and median

df_train, df_test = train_test_split(classification_df, test_size=0.2, random_state=1234)
df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=1234)

ds_train = Dataset.from_pandas(df_train)
ds_val = Dataset.from_pandas(df_val)
ds_test = Dataset.from_pandas(df_test)


print('Tokenizing...')
config = AutoConfig.from_pretrained('distilbert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('./tokenizers/codonBERT', model_max_length=trunc_len, padding_side='left', truncation_side='right')


tokenized_ds_train = ds_train.map(lambda d : tokenizer(d['text'], truncation=True, padding=True), batched=True)
tokenized_ds_val = ds_val.map(lambda d : tokenizer(d['text'], truncation=True, padding=True), batched=True)
tokenized_ds_test = ds_test.map(lambda d : tokenizer(d['text'], truncation=True, padding=True), batched=True)

print('Building Model...')
#model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model = AutoModelForSequenceClassification.from_config(config, num_labels=3) #randomly initialize it

training_args = TrainingArguments(
    output_dir='./models/codonBERT-multi_{}'.format(RUN),
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=50,
    weight_decay=0.01,
)

trainer = Trainer(
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
with open('./results/codonBERT-multi_{}.txt'.format(RUN),'w') as data: 
      data.write(str(scores))

