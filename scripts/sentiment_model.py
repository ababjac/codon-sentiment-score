import helpers

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras as ks
from tensorflow.keras import layers

def create_model(lstm_units=1, neurons_dense=1, dropout_rate=0.1, embedding_size=2, max_text_len=helpers.VOCAB_SIZE, learning_rate=0.5):
    model = ks.Sequential()
    model.add(layers.Embedding(helpers.VOCAB_SIZE, embedding_size))
    model.add(layers.LSTM(units=lstm_units))
    model.add(layers.Dense(neurons_dense, activation="relu"))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(3, activation="softmax"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

############ SIMPLE TESTING ###############
# model = create_model(max_text_len=3)
# model.fit([[1,2,3], [2,3,4]], [0, 1])
# pred = model.predict([[1,2,3], [2,3,4]])
#
# print(pred)
