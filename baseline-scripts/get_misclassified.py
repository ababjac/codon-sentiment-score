import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import helpers

df = pd.read_csv('data/ecoli_complete.csv')
seq = np.fromfile('results/testR-binary_1.csv', sep=',')

df = helpers.add_codons_to_df(df, 'mRNA_cleaned')
df['sentiment'] = np.where(df['abundance'] > np.median(df['abundance'].values), 1, 0)

df_train, df_test = train_test_split(df, test_size=0.2, random_state=62835) #save 20% for testing

#print(df_test.columns)
#print(seq)
out_df = pd.DataFrame()
out_df['gene'] = df_test['Gene']
out_df['y_true'] = df_test['sentiment']
out_df['y_pred'] = seq

out_df[out_df['y_true'] != out_df['y_pred']].to_csv('results/binary-test_misclassified_1.csv')
#out_df['misclass'] = np.where(out_df['y_true'] != out_df['y_pred'], 1, 0)
#out_df['confidence_score'] = df_test['confidence_score']

#print(np.median(out_df.loc[out_df['misclass'] == 1, 'confidence_score']))
#print(np.median(out_df.loc[out_df['misclass'] == 0, 'confidence_score']))
