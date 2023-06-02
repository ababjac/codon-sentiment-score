#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
#from transformers import AutoTokenizer, AutoModel, TFAutoModelForSequenceClassification
#from bertviz import model_view, head_view
#from alibi.explainers import IntegratedGradients
import json
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import torch
map_location=torch.device('cpu')


# In[4]:


# read the json file
with open('./models/codonBERT-binary_reg_ecoli-nopre/checkpoint-4150/trainer_state.json', 'r') as f:
    data = json.load(f)

# create a pandas dataframe from the log_history
df = pd.DataFrame(data['log_history'])
epochs = df['epoch']
losses = df['loss']
print(df)

# plot the data using seaborn
#sns.set_style("darkgrid")
#fig, ax = plt.subplots(figsize=(8, 6))
#sns.lineplot(x='epoch', y='loss', data=df)
#ax.set_xlim([0, max(epochs)])
#ax.set_ylim([0, max(losses) + 0.01])

plt.plot(df['epoch'], df['eval_loss'])
plt.plot(df['epoch'], df['loss'])

plt.title('Training Loss over Epochs For E. Coli')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
plt.savefig('images/ecoli_loss_plot.png')

