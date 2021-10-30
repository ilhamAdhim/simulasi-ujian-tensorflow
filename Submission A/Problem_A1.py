# =================================================================================
# PROBLEM A1
#
# Given two arrays, train a neural network model to match the X to the Y.
# Predict the model with new values of X [-2.0, 10.0]
# We provide the model prediction, do not change the code.
#
# The test infrastructure expects a trained model that accepts
# an input shape of [1].
# Do not use lambda layers in your model.
#
# Desired loss (MSE) < 1e-4
# -----------------------------------
# Muhammad Ilham Adhim
# =================================================================================


import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping


def solution_A1():
    X = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0,
                  2.0, 3.0, 4.0, 5.0], dtype=float)
    Y = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                  12.0, 13.0, 14.0, ], dtype=float)

    # YOUR CODE HERE
    class MyCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('loss') < 1e-4):
                print('\nMSE Loss is lesser than 1e-4 !')
                print('\nStop training model...')
                self.model.stop_training = True

    # Inisiasi class MyCallback ke dalam variable callbacks
    custom_callback = MyCallback()

    model = Sequential([
        Dense(units=1, input_shape=[1])
    ])

    model.compile(optimizer='sgd',
                  metrics='accuracy',
                  loss='mean_squared_error')

    model.fit(X, Y, epochs=500, callbacks=[custom_callback])

    print(model.predict([-2.0, 10.0]))
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_A1()
    model.save("model_A1.h5")