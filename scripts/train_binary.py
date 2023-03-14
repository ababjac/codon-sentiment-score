import helpers
import sentiment_model
import graph

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scikeras.wrappers import KerasClassifier, KerasRegressor
from keras.utils import np_utils

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_objective, plot_histogram
from skopt.callbacks import VerboseCallback

from alibi.explainers import IntegratedGradients

print('Reading data...')
df = pd.read_csv('data/ecoli_complete.csv', index_col=0)

print('Processing data...')
df = helpers.add_codons_to_df(df, 'mRNA_cleaned')
df['sentiment'] = np.where(df['abundance'] > np.median(df['abundance'].values), 1, 0)

MAX = max([len(elem) for elem in df['codons_cleaned']]) #get max sequence length for padding

df_train, df_test = train_test_split(df, test_size=0.2, random_state=62835) #save 20% for testing

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
param_grid = {
    'learning_rate' : Real(0.0001, 0.5, prior='log-uniform'),
    'dropout_rate' : Real(0.1, 0.5, prior='log-uniform'),
    'lstm_units' : Integer(1, 10),
    'neurons_dense' : Integer(1, 300),
    'embedding_size' : Integer(2, 500)
}

model = KerasClassifier(model=sentiment_model.create_model, epochs=10, verbose=1, validation_split=0.2, lstm_units=1, neurons_dense=1, dropout_rate=0.1, embedding_size=2, max_text_len=helpers.VOCAB_SIZE, learning_rate=0.5)

grid = BayesSearchCV(
    estimator=model,
    search_spaces=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=94722).split(X_train, y_train),
    verbose=True,
    scoring='roc_auc',
    n_jobs=8
)

result = grid.fit(X_train_seq, y_train)
print('Best params: ', grid.best_params_)
params = grid.best_params_

for i in range(len(grid.optimizer_results_)):
    plot_objective(grid.optimizer_results_[i])
    plt.savefig('images/grid/optimizer_results_binary_{}.png'.format(i))
    plt.close()

print('Print predicting with best params...')
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_seq)

out_array = np.array(y_pred)
out_array.tofile('results/testR-binary.csv', sep=',')

print('Plotting...')
graph.plot_confusion_matrix_binary(y_pred=y_pred, y_actual=y_test, title='Expression Classification', filename='images/confusion-matrix/CM-binary.png')

test_loss, test_auc, test_acc, test_precision, test_recall = best_model.model_.evaluate(X_test_seq, y_test)
print('Binary Results:')
print('Accuracy:', test_acc)
print('Precision:', test_precision)
print('Recall:', test_recall)
print('AUC:', test_auc)
print('Loss:', test_loss)
