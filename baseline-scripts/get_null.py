import os
import pandas as pd
import numpy as np
import helpers
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

PATH='/lustre/isaac/proj/UTK0196/codon-expression-data/fullTableForTrainning/'
'''
filelist = os.listdir(PATH)
df_list = [pd.read_csv(PATH+file) for file in filelist]
df = pd.concat(df_list)

df = helpers.add_codons_to_df(df, 'Sequence')
df['sentiment'] = np.where(df['median_exp'] > np.median(df['median_exp'].values), 1, 0)

null = np.zeros(len(df))

print("Accuracy:", accuracy_score(df.sentiment, null))
print("Precision:", precision_score(df.sentiment, null))
print("Recall:", recall_score(df.sentiment, null))
print("AUC:", roc_auc_score(df.sentiment, null))
'''


filelist = os.listdir(PATH)
df_list = [pd.read_csv(PATH+file) for file in filelist]
df = pd.concat(df_list)

df = helpers.add_codons_to_df(df, 'Sequence')
low, high = df.median_exp.quantile([0.33, 0.67])
high_l = np.where(df['median_exp'] > high, 2, 0)
low_l = np.where(df['median_exp'] > low, 0, 1)
df['sentiment'] = high_l+low_l

null = np.ones(len(df))
null_probs = np.array([[0.0, 0.0, 1.0] for _ in range(len(df))])
#print(null_probs)

print("Accuracy:", accuracy_score(df.sentiment, null))
print("Precision:", precision_score(df.sentiment, null, average='macro'))
print("Recall:", recall_score(df.sentiment, null, average='macro'))
print("AUC:", roc_auc_score(df.sentiment, null_probs, multi_class='ovr'))

