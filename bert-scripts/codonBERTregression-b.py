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
from scipy.stats import spearmanr

PATH='/lustre/isaac/proj/UTK0196/codon-expression-data/fullTableForTrainning/'
RUN=1
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def compute_metrics(epred):
    print('in compute_metrics')
    # Computes metrics from specialized output from huggingface
    #print(epred)

    logits = epred[0]
    labels = epred[1].reshape(-1,1)

    print(logits.shape, labels.shape)
    metrics = {}

    metrics['mse'] = mean_squared_error(labels, logits)
    metrics['mae'] = mean_absolute_error(labels, logits)
    metrics['r2'] = r2_score(labels, logits)
    rho, pval = spearmanr(labels, logits)
    metrics['spearmanr'] = rho
    #metrics['single_squared_errors'] = ((logits - labels).flatten()**2).tolist()

    return metrics

# class BertRegresser(DistilBertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.bert = DistilBertModel(config)
#         #The output layer that takes the [CLS] representation and gives an output
#         self.cls_layer1 = torch.nn.Linear(config.hidden_size, 1)
#         #self.cls_layer1 = torch.nn.Linear(config.hidden_size,128)
#         #self.relu1 = torch.nn.ReLU()
#         #self.ff1 = torch.nn.Linear(128,128)
#         #self.tanh1 = torch.nn.Tanh()
#         #self.ff2 = torch.nn.Linear(128,1)

#     def forward(self, input_ids, attention_mask, labels):
#         #Feed the input to Bert model to obtain contextualized representations
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         #Obtain the representations of [CLS] heads
#         #logits = outputs.last_hidden_state[:,0,:]
#         output = self.cls_layer1(outputs.last_hidden_state[:,0,:])
#         #output = self.relu1(output)
#         #output = self.ff1(output)
#         #output = self.tanh1(output)
#         #output = self.ff2(output)
#         return output


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        print('in compute_loss')

        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        print(outputs)
        
        logits = outputs.logits

        print(labels.shape, logits.shape)

        # compute custom loss
        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(labels.float(), logits)
        return (loss, outputs) if return_outputs else loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Tells the model we need to use the GPU

print('Reading data...')
filelist = os.listdir(PATH) 
df_list = [pd.read_csv(PATH+file) for file in filelist]
df = pd.concat(df_list)

s = []
for file, _df in zip(filelist, df_list):
    species = file.partition('_')[0]
    l = [species]*len(_df)
    s.extend(l)
    
df['species'] = s

SPECIES = 'atha'

df = df[df['species'] == SPECIES] #train on only yeast sequences

df = helpers.add_codons_to_df(df, 'Sequence')
labels = df['median_exp']

#labels = labels.type(torch.LongTensor)

classification_df = pd.DataFrame({'text' : df['codons_cleaned'], 'label' : labels})
#MAX = int(max([(len(elem) / 3) for elem in df['codons_cleaned']])) #get max sequence length for padding
#MED = int(np.median([(len(elem) / 3) for elem in df['codons_cleaned']])) #get median sequence length for padding
#print(MED)
#trunc_len = int((MAX + MED) / 2) #set truncation somewhere between max and median
trunc_len = 512

df_train, df_test = train_test_split(classification_df, test_size=0.2, random_state=1234)
df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=1234)

#print(len(df_val))

ds_train = Dataset.from_pandas(df_train)
ds_val = Dataset.from_pandas(df_val)
ds_test = Dataset.from_pandas(df_test)

del classification_df
del df_train
del df_test
del df_val

print('Tokenizing...')
config = AutoConfig.from_pretrained('distilbert-base-uncased', max_position_embeddings=trunc_len)
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
#model = AutoModelForSequenceClassification.from_pretrained('./models/codonBERT-binary_1/checkpoint-127000', num_labels=1, ignore_mismatched_sizes=True)
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
#model = AutoModelForSequenceClassification.from_config(config) #randomly initialize it

## Creating the model from the desired transformer model
#model = BertRegresser.from_pretrained('./models/codonBERT-binary_1/checkpoint-127000')#, config=config)
#model = BertRegresser.from_pretrained('distilbert-base-uncased', config=config)
#optimizer = torch.optim.Adam(params=model.parameters())

training_args = TrainingArguments(
    output_dir='./models/codonBERT_reg_{}-nopre'.format(SPECIES),
    learning_rate=1e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=50,
    weight_decay=0.01,
    optim="adamw_torch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="spearmanr",
    load_best_model_at_end=True,
    remove_unused_columns=True,
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds_train,
    eval_dataset=tokenized_ds_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


print('Training...')
#out = trainer.predict(tokenized_ds_val)
#logits, labels, metrics = out
#print(logits.shape, labels.shape)
trainer.train()
trainer.evaluate()
out = trainer.predict(test_dataset=tokenized_ds_test)

scores = compute_metrics(out)
with open('./results/codonBERT_reg_scores_{}-nopre.txt'.format(SPECIES),'w') as data: 
    data.write(str(scores))

with open('./results/codonBERT_reg_output_{}-nopre.txt'.format(SPECIES),'w') as data:
    for val in out[0]:
        data.write(str(val)+'\n')

