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
        #print(outputs)
        
        logits = outputs.logits

        print(labels.shape, logits.shape)

        # compute custom loss
        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(labels.float(), logits)
        return (loss, outputs) if return_outputs else loss

#https://stackoverflow.com/questions/67689219/copy-one-layers-weights-from-one-huggingface-bert-model-to-another
def setLayers(t, s, parts):
    target = dict(t.named_parameters())
    source = dict(s.named_parameters())

    #print(any('bert.embeddings.word_embeddings.weight' for val in source.keys()))

    for part in parts:
        target[part].data.copy_(source[part].data)  
        target[part].requires_grad = False

parts = [
        'bert.embeddings.word_embeddings.weight',
        'bert.embeddings.position_embeddings.weight',
        'bert.embeddings.token_type_embeddings.weight',
        'bert.embeddings.LayerNorm.weight',
        'bert.embeddings.LayerNorm.bias',
        'bert.encoder.layer.0.attention.self.query.weight',
        'bert.encoder.layer.0.attention.self.query.bias',
        'bert.encoder.layer.0.attention.self.key.weight',
        'bert.encoder.layer.0.attention.self.key.bias',
        'bert.encoder.layer.0.attention.self.value.weight',
        'bert.encoder.layer.0.attention.self.value.bias',
        'bert.encoder.layer.0.attention.output.dense.weight',
        'bert.encoder.layer.0.attention.output.dense.bias',
        'bert.encoder.layer.0.attention.output.LayerNorm.weight',
        'bert.encoder.layer.0.attention.output.LayerNorm.bias',
        'bert.encoder.layer.0.intermediate.dense.weight',
        'bert.encoder.layer.0.intermediate.dense.bias',
        'bert.encoder.layer.0.output.dense.weight',
        'bert.encoder.layer.0.output.dense.bias',
        'bert.encoder.layer.0.output.LayerNorm.weight',
        'bert.encoder.layer.0.output.LayerNorm.bias',
        'bert.encoder.layer.1.attention.self.query.weight',
        'bert.encoder.layer.1.attention.self.query.bias',
        'bert.encoder.layer.1.attention.self.key.weight',
        'bert.encoder.layer.1.attention.self.key.bias',
        'bert.encoder.layer.1.attention.self.value.weight',
        'bert.encoder.layer.1.attention.self.value.bias',
        'bert.encoder.layer.1.attention.output.dense.weight',
        'bert.encoder.layer.1.attention.output.dense.bias',
        'bert.encoder.layer.1.attention.output.LayerNorm.weight',
        'bert.encoder.layer.1.attention.output.LayerNorm.bias',
        'bert.encoder.layer.1.intermediate.dense.weight',
        'bert.encoder.layer.1.intermediate.dense.bias',
        'bert.encoder.layer.1.output.dense.weight',
        'bert.encoder.layer.1.output.dense.bias',
        'bert.encoder.layer.1.output.LayerNorm.weight',
        'bert.encoder.layer.1.output.LayerNorm.bias',
        'bert.encoder.layer.2.attention.self.query.weight',
        'bert.encoder.layer.2.attention.self.query.bias',
        'bert.encoder.layer.2.attention.self.key.weight',
        'bert.encoder.layer.2.attention.self.key.bias',
        'bert.encoder.layer.2.attention.self.value.weight',
        'bert.encoder.layer.2.attention.self.value.bias',
        'bert.encoder.layer.2.attention.output.dense.weight',
        'bert.encoder.layer.2.attention.output.dense.bias',
        'bert.encoder.layer.2.attention.output.LayerNorm.weight',
        'bert.encoder.layer.2.attention.output.LayerNorm.bias',
        'bert.encoder.layer.2.intermediate.dense.weight',
        'bert.encoder.layer.2.intermediate.dense.bias',
        'bert.encoder.layer.2.output.dense.weight',
        'bert.encoder.layer.2.output.dense.bias',
        'bert.encoder.layer.2.output.LayerNorm.weight',
        'bert.encoder.layer.2.output.LayerNorm.bias',
        'bert.encoder.layer.3.attention.self.query.weight',
        'bert.encoder.layer.3.attention.self.query.bias',
        'bert.encoder.layer.3.attention.self.key.weight',
        'bert.encoder.layer.3.attention.self.key.bias',
        'bert.encoder.layer.3.attention.self.value.weight',
        'bert.encoder.layer.3.attention.self.value.bias',
        'bert.encoder.layer.3.attention.output.dense.weight',
        'bert.encoder.layer.3.attention.output.dense.bias',
        'bert.encoder.layer.3.attention.output.LayerNorm.weight',
        'bert.encoder.layer.3.attention.output.LayerNorm.bias',
        'bert.encoder.layer.3.intermediate.dense.weight',
        'bert.encoder.layer.3.intermediate.dense.bias',
        'bert.encoder.layer.3.output.dense.weight',
        'bert.encoder.layer.3.output.dense.bias',
        'bert.encoder.layer.3.output.LayerNorm.weight',
        'bert.encoder.layer.3.output.LayerNorm.bias',
        'bert.encoder.layer.4.attention.self.query.weight',
        'bert.encoder.layer.4.attention.self.query.bias',
        'bert.encoder.layer.4.attention.self.key.weight',
        'bert.encoder.layer.4.attention.self.key.bias',
        'bert.encoder.layer.4.attention.self.value.weight',
        'bert.encoder.layer.4.attention.self.value.bias',
        'bert.encoder.layer.4.attention.output.dense.weight',
        'bert.encoder.layer.4.attention.output.dense.bias',
        'bert.encoder.layer.4.attention.output.LayerNorm.weight',
        'bert.encoder.layer.4.attention.output.LayerNorm.bias',
        'bert.encoder.layer.4.intermediate.dense.weight',
        'bert.encoder.layer.4.intermediate.dense.bias',
        'bert.encoder.layer.4.output.dense.weight',
        'bert.encoder.layer.4.output.dense.bias',
        'bert.encoder.layer.4.output.LayerNorm.weight',
        'bert.encoder.layer.4.output.LayerNorm.bias',
        'bert.encoder.layer.5.attention.self.query.weight',
        'bert.encoder.layer.5.attention.self.query.bias',
        'bert.encoder.layer.5.attention.self.key.weight',
        'bert.encoder.layer.5.attention.self.key.bias',
        'bert.encoder.layer.5.attention.self.value.weight',
        'bert.encoder.layer.5.attention.self.value.bias',
        'bert.encoder.layer.5.attention.output.dense.weight',
        'bert.encoder.layer.5.attention.output.dense.bias',
        'bert.encoder.layer.5.attention.output.LayerNorm.weight',
        'bert.encoder.layer.5.attention.output.LayerNorm.bias',
        'bert.encoder.layer.5.intermediate.dense.weight',
        'bert.encoder.layer.5.intermediate.dense.bias',
        'bert.encoder.layer.5.output.dense.weight',
        'bert.encoder.layer.5.output.dense.bias',
        'bert.encoder.layer.5.output.LayerNorm.weight',
        'bert.encoder.layer.5.output.LayerNorm.bias',
        'bert.encoder.layer.6.attention.self.query.weight',
        'bert.encoder.layer.6.attention.self.query.bias',
        'bert.encoder.layer.6.attention.self.key.weight',
        'bert.encoder.layer.6.attention.self.key.bias',
        'bert.encoder.layer.6.attention.self.value.weight',
        'bert.encoder.layer.6.attention.self.value.bias',
        'bert.encoder.layer.6.attention.output.dense.weight',
        'bert.encoder.layer.6.attention.output.dense.bias',
        'bert.encoder.layer.6.attention.output.LayerNorm.weight',
        'bert.encoder.layer.6.attention.output.LayerNorm.bias',
        'bert.encoder.layer.6.intermediate.dense.weight',
        'bert.encoder.layer.6.intermediate.dense.bias',
        'bert.encoder.layer.6.output.dense.weight',
        'bert.encoder.layer.6.output.dense.bias',
        'bert.encoder.layer.6.output.LayerNorm.weight',
        'bert.encoder.layer.6.output.LayerNorm.bias',
        'bert.encoder.layer.7.attention.self.query.weight',
        'bert.encoder.layer.7.attention.self.query.bias',
        'bert.encoder.layer.7.attention.self.key.weight',
        'bert.encoder.layer.7.attention.self.key.bias',
        'bert.encoder.layer.7.attention.self.value.weight',
        'bert.encoder.layer.7.attention.self.value.bias',
        'bert.encoder.layer.7.attention.output.dense.weight',
        'bert.encoder.layer.7.attention.output.dense.bias',
        'bert.encoder.layer.7.attention.output.LayerNorm.weight',
        'bert.encoder.layer.7.attention.output.LayerNorm.bias',
        'bert.encoder.layer.7.intermediate.dense.weight',
        'bert.encoder.layer.7.intermediate.dense.bias',
        'bert.encoder.layer.7.output.dense.weight',
        'bert.encoder.layer.7.output.dense.bias',
        'bert.encoder.layer.7.output.LayerNorm.weight',
        'bert.encoder.layer.7.output.LayerNorm.bias',
        'bert.encoder.layer.8.attention.self.query.weight',
        'bert.encoder.layer.8.attention.self.query.bias',
        'bert.encoder.layer.8.attention.self.key.weight',
        'bert.encoder.layer.8.attention.self.key.bias',
        'bert.encoder.layer.8.attention.self.value.weight',
        'bert.encoder.layer.8.attention.self.value.bias',
        'bert.encoder.layer.8.attention.output.dense.weight',
        'bert.encoder.layer.8.attention.output.dense.bias',
        'bert.encoder.layer.8.attention.output.LayerNorm.weight',
        'bert.encoder.layer.8.attention.output.LayerNorm.bias',
        'bert.encoder.layer.8.intermediate.dense.weight',
        'bert.encoder.layer.8.intermediate.dense.bias',
        'bert.encoder.layer.8.output.dense.weight',
        'bert.encoder.layer.8.output.dense.bias',
        'bert.encoder.layer.8.output.LayerNorm.weight',
        'bert.encoder.layer.8.output.LayerNorm.bias',
        'bert.encoder.layer.9.attention.self.query.weight',
        'bert.encoder.layer.9.attention.self.query.bias',
        'bert.encoder.layer.9.attention.self.key.weight',
        'bert.encoder.layer.9.attention.self.key.bias',
        'bert.encoder.layer.9.attention.self.value.weight',
        'bert.encoder.layer.9.attention.self.value.bias',
        'bert.encoder.layer.9.attention.output.dense.weight',
        'bert.encoder.layer.9.attention.output.dense.bias',
        'bert.encoder.layer.9.attention.output.LayerNorm.weight',
        'bert.encoder.layer.9.attention.output.LayerNorm.bias',
        'bert.encoder.layer.9.intermediate.dense.weight',
        'bert.encoder.layer.9.intermediate.dense.bias',
        'bert.encoder.layer.9.output.dense.weight',
        'bert.encoder.layer.9.output.dense.bias',
        'bert.encoder.layer.9.output.LayerNorm.weight',
        'bert.encoder.layer.9.output.LayerNorm.bias',
        'bert.encoder.layer.10.attention.self.query.weight',
        'bert.encoder.layer.10.attention.self.query.bias',
        'bert.encoder.layer.10.attention.self.key.weight',
        'bert.encoder.layer.10.attention.self.key.bias',
        'bert.encoder.layer.10.attention.self.value.weight',
        'bert.encoder.layer.10.attention.self.value.bias',
        'bert.encoder.layer.10.attention.output.dense.weight',
        'bert.encoder.layer.10.attention.output.dense.bias',
        'bert.encoder.layer.10.attention.output.LayerNorm.weight',
        'bert.encoder.layer.10.attention.output.LayerNorm.bias',
        'bert.encoder.layer.10.intermediate.dense.weight',
        'bert.encoder.layer.10.intermediate.dense.bias',
        'bert.encoder.layer.10.output.dense.weight',
        'bert.encoder.layer.10.output.dense.bias',
        'bert.encoder.layer.10.output.LayerNorm.weight',
        'bert.encoder.layer.10.output.LayerNorm.bias',
        'bert.encoder.layer.11.attention.self.query.weight',
        'bert.encoder.layer.11.attention.self.query.bias',
        'bert.encoder.layer.11.attention.self.key.weight',
        'bert.encoder.layer.11.attention.self.key.bias',
        'bert.encoder.layer.11.attention.self.value.weight',
        'bert.encoder.layer.11.attention.self.value.bias',
        'bert.encoder.layer.11.attention.output.dense.weight',
        'bert.encoder.layer.11.attention.output.dense.bias',
        'bert.encoder.layer.11.attention.output.LayerNorm.weight',
        'bert.encoder.layer.11.attention.output.LayerNorm.bias',
        'bert.encoder.layer.11.intermediate.dense.weight',
        'bert.encoder.layer.11.intermediate.dense.bias',
        'bert.encoder.layer.11.output.dense.weight',
        'bert.encoder.layer.11.output.dense.bias',
        'bert.encoder.layer.11.output.LayerNorm.weight',
        'bert.encoder.layer.11.output.LayerNorm.bias',
]

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

SPECIES = 'ecoli'

df = df[df['species'] == SPECIES] #train on only yeast sequences

df = helpers.add_codons_to_df(df, 'Sequence')
labels = np.log(df['median_exp'])

#labels = labels.type(torch.LongTensor)

classification_df = pd.DataFrame({'text' : df['codons_cleaned'], 'label' : labels})
#MAX = int(max([(len(elem) / 3) for elem in df['codons_cleaned']])) #get max sequence length for padding
#MED = int(np.median([(len(elem) / 3) for elem in df['codons_cleaned']])) #get median sequence length for padding
#print(MED)
#trunc_len = int((MAX + MED) / 2) #set truncation somewhere between max and median
trunc_len = 1064

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
config = AutoConfig.from_pretrained('bert-base-uncased', max_position_embeddings=trunc_len, num_labels=1)
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
pretrained_model = AutoModelForSequenceClassification.from_pretrained('./models/codonBERT-binary-large_1/checkpoint-127330')
model = AutoModelForSequenceClassification.from_config(config)

setLayers(model, pretrained_model, parts) #setting weights from pretrained binary classifier except for last layers

#for name, param in pretrained_model.named_parameters():
#    print(name, param.requires_grad)

#model = AutoModelForSequenceClassification.from_config(config) #randomly initialize it

## Creating the model from the desired transformer model
#model = BertRegresser.from_pretrained('./models/codonBERT-binary_1/checkpoint-127000')#, config=config)
#model = BertRegresser.from_pretrained('distilbert-base-uncased', config=config)
#optimizer = torch.optim.Adam(params=model.parameters())

training_args = TrainingArguments(
    output_dir='./models/codonBERT_binary_reg_{}-pre-frozen-norm'.format(SPECIES),
    learning_rate=1e-6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=200,
    weight_decay=0.001,
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


#print('Training...')
trainer.train()
trainer.evaluate()
out = trainer.predict(test_dataset=tokenized_ds_test)
logits, labels, metrics = out

#print(metrics)
#print(logits)
#print(logits.shape, labels.shape)
#scores = compute_metrics(out)

with open('./results/codonBERT_binary_reg_scores_{}-pre-frozen-norm.txt'.format(SPECIES),'w') as data: 
    data.write(str(metrics))

with open('./results/codonBERT_binary_reg_output_{}-pre-frozen-norm.txt'.format(SPECIES),'w') as data:
    for val in logits:
        data.write(str(val)+'\n')

