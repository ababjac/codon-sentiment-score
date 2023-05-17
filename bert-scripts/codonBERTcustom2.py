#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
import pickle
import os
import gc
import helpers

from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.nn import softmax
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split

PATH='/lustre/isaac/proj/UTK0196/codon-expression-data/fullTableForTrainning/'
RUN=1
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def compute_metrics(epred):
    # Computes metrics from specialized output from huggingface
    logits = epred[0]
    probs = softmax(logits)
    preds = np.argmax(logits, axis=-1)
    labels = epred[1]

    metrics = {}
    #metrics['auprc'] = average_precision_score(labels, preds[:,1])
    metrics['auroc'] = roc_auc_score(labels, probs[:,1], multi_class='ovr', average='micro')
    metrics['accuracy'] = CategoricalAccuracy(labels, logits)
    metrics['precision'] = precision_score(labels, preds[:, 1], average='micro')
    metrics['recall'] = recall_score(labels, preds[:, 1], average='micro')
        
    return metrics




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Tells the model we need to use the GPU

print('Reading data...')
filelist = os.listdir(PATH) 
df_list = [pd.read_csv(PATH+file) for file in filelist]
df = pd.concat(df_list)

df = helpers.add_codons_to_df(df, 'Sequence')
low, high = df.median_exp.quantile([0.33, 0.67])
high_l = np.where(df['median_exp'] > high, 2, 0)
low_l = np.where(df['median_exp'] > low, 0, 1)
labels = high_l+low_l


classification_df = pd.DataFrame({'text' : df['codons_cleaned'], 'label' : labels})
#MAX = int(max([(len(elem) / 3) for elem in df['codons_cleaned']])) #get max sequence length for padding
#MED = int(np.median([(len(elem) / 3) for elem in df['codons_cleaned']])) #get median sequence length for padding

#trunc_len = int((MAX + MED) / 2) #set truncation somewhere between max and median
trunc_len=1024

df_train, df_test = train_test_split(classification_df, test_size=0.2, random_state=1234)
df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=1234)

del classification_df

ds_train = Dataset.from_pandas(df_train)
ds_val = Dataset.from_pandas(df_val)
ds_test = Dataset.from_pandas(df_test)

del df_train
del df_val
del df_test

print('Tokenizing...')
config = AutoConfig.from_pretrained('distilbert-base-uncased', max_position_embeddings=trunc_len, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained('./tokenizers/codonBERT', model_max_length=trunc_len, padding_side='left', truncation_side='right')

tokenized_ds_train = ds_train.map(lambda d : tokenizer(d['text'], truncation=True, padding=True), batched=True)
tokenized_ds_val = ds_val.map(lambda d : tokenizer(d['text'], truncation=True, padding=True), batched=True)
tokenized_ds_test = ds_test.map(lambda d : tokenizer(d['text'], truncation=True, padding=True), batched=True)

del ds_train
del ds_val
del ds_test
gc.collect()

torch.cuda.empty_cache()

print('Building Model...')
model = AutoModelForSequenceClassification.from_pretrained('./models/codonBERT-multi_0/checkpoint-527500', num_labels=3)
#model = AutoModelForSequenceClassification.from_config(config) #randomly initialize it

training_args = TrainingArguments(
    output_dir='./models/codonBERT-multi_{}'.format(RUN),
    learning_rate=2e-6,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
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

