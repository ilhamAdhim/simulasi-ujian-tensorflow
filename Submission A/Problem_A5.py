import csv
import tensorflow as tf
import numpy as np
import urllib
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM

# DO NOT CHANGE THIS CODE


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def solution_A5():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sunspots.csv'
    urllib.request.urlretrieve(data_url, 'sunspots.csv')

    time_step = []
    sunspots = []

    with open('sunspots.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            sunspots.append(float(row[2]))
            time_step.append(int(row[0]))

    series = np.array(sunspots)
    time = np.array(time_step)

    # Normalization Function. DO NOT CHANGE THIS CODE
    min = np.min(series)
    max = np.max(series)
    series -= min
    series /= max
    time = np.array(time_step)

    # DO NOT CHANGE THIS CODE
    split_time = 3000

    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]

    # DO NOT CHANGE THIS CODE
    window_size = 30
    batch_size = 32
    shuffle_buffer_size = 1000

    train_set = windowed_dataset(x_train, window_size=window_size,
                                 batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)

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
            if (logs.get('mae') < 0.085):
                print('\nMAE is lesser than 0.085 !')
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
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.001, momentum=0.8), metrics='mae')

    model.fit(train_set,
              epochs=100,
              verbose=1,
              callbacks=[custom_callback, early_stopping_callbacks])

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_A5()
    model.save("model_A5.h5")
