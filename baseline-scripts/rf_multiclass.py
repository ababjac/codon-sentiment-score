import helpers
#import sentiment_model
import graph
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from scikeras.wrappers import KerasClassifier, KerasRegressor

PATH='/lustre/isaac/proj/UTK0196/codon-expression-data/fullTableForTrainning/'


print('Reading data...')
#df = pd.read_csv('data/ecoli_complete.csv', index_col=0)

print('Processing data...')
#df = helpers.add_codons_to_df(df, 'mRNA_cleaned')
#low, high = df.abundance.quantile([0.33, 0.67])
#high_l = np.where(df['abundance'] > high, 1, 0)
#low_l = np.where(df['abundance'] > low, 0, -1)
#df['sentiment'] = high_l+low_l

filelist = os.listdir(PATH)
df_list = [pd.read_csv(PATH+file) for file in filelist]
df = pd.concat(df_list)

df = helpers.add_codons_to_df(df, 'Sequence')
low, high = df.median_exp.quantile([0.33, 0.67])
high_l = np.where(df['median_exp'] > high, 2, 0)
low_l = np.where(df['median_exp'] > low, 0, 1)
df['sentiment'] = high_l+low_l

#chosen_idx = np.random.choice(len(df), replace=False, size=1000)
#df = df.iloc[chosen_idx]

MAX = max([len(elem) for elem in df['codons_cleaned']]) #get max sequence length for padding

df_train, df_test = train_test_split(df, test_size=0.2) #save 20% for testing

X_train = df_train['codons_cleaned'].values
y_train = df_train['sentiment'].values

X_test = df_test['codons_cleaned'].values
y_test = df_test['sentiment'].values

X_full = df['codons_cleaned'].values

print('Tokenizing...')
tokenizer = Tokenizer(num_words=helpers.VOCAB_SIZE)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_full_seq = tokenizer.texts_to_sequences(X_full)

X_train_seq = pad_sequences(X_train_seq, MAX)
X_test_seq = pad_sequences(X_test_seq, MAX)
X_full_seq = pad_sequences(X_full_seq, MAX)

print('Building Model...')
param_grid = {'min_weight_fraction_leaf': [0.1, 0.3, 0.5],
               'bootstrap': [False],
               'max_depth': [20, 50],
               'max_leaf_nodes': [10, 30, 50],
               'n_estimators': [100]}

clf = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=94722).split(X_train, y_train),
        n_jobs=-1,
        verbose=3
    )

print('Building model...')
clf.fit(X_train_seq, y_train)

print('Predicting on test data...')
y_pred = clf.predict(X_test_seq)
y_prob = clf.predict_proba(X_test_seq) #get probabilities for AUC
probs = y_prob[:,1]
#print(y_prob)

print('Calculating metrics for RF multi...')
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("AUC:", roc_auc_score(y_test, y_prob, multi_class='ovr'))
