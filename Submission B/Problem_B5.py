# ============================================================================================
# PROBLEM B5
#
# Build and train a neural network model using the Daily Max Temperature.csv dataset.
# Use MAE as the metrics of your neural network model.
# We provided code for normalizing the data. Please do not change the code.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is downloaded from https://github.com/jbrownlee/Datasets
#
# Desired MAE < 0.2 on the normalized dataset.
# ------------------------
# Muhammad Ilham Adhim
# ============================================================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import urllib

from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import SGD


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def solution_B5():
    data_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-max-temperatures.csv'
    urllib.request.urlretrieve(data_url, 'daily-max-temperatures.csv')

    time_step = []
    temps = []

    with open('daily-max-temperatures.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        step = 0
        for row in reader:
            temps.append(float(row[1]))
            time_step.append(step)
            step = step + 1

        series = np.array(temps)  # YOUR CODE HERE

        # Normalization Function. DO NOT CHANGE THIS CODE
        min = np.min(series)
        max = np.max(series)
        series -= min
        series /= max
        time = np.array(time_step)

        # DO NOT CHANGE THIS CODE
        split_time = 2500

        time_train = time[:split_time]
        x_train = series[:split_time]
        time_valid = time[split_time:]
        x_valid = series[split_time:]

        # DO NOT CHANGE THIS CODE
        window_size = 64
        batch_size = 256
        shuffle_buffer_size = 1000

        train_set = windowed_dataset(
            x_train, window_size, batch_size, shuffle_buffer_size)
        print(train_set)
        print(x_train.shape)

        model = tf.keras.models.Sequential([
            # YOUR CODE HERE.
            LSTM(64, return_sequences=True,  input_shape=[None, 1]),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(8, activation="relu"),
            Dense(1)
        ])

        # Create callback function to stop data training once the target MAE has been fulfilled
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('mae') < 0.2):
                print('\nMAE is lesser than 0.2 !')
                print('\nStop training model...')
                self.model.stop_training = True

    # Inisiasi class myCallback ke dalam variable callbacks
    custom_callback = myCallback()

    early_stopping_callbacks = EarlyStopping(
        monitor="mae",
        min_delta=0.01,
        patience=5,
        restore_best_weights=True,
    )

    # YOUR CODE
    model.compile(loss="mse", optimizer=SGD(
        learning_rate=0.001, momentum=0.8), metrics='mae')

    model.fit(train_set,
              epochs=100,
              verbose=1,
              callbacks=[custom_callback, early_stopping_callbacks])

    # YOUR CODE HERE
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_B5()
    model.save("model_B5.h5")
