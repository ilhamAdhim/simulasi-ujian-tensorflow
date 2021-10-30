
# =====================================================================================
# PROBLEM A2
#
# Build a Neural Network Model for Horse or Human Dataset.
# The test will expect it to classify binary classes.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy and validation_accuracy > 83%
# -----------------------------------
# Muhammad Ilham Adhim
# ======================================================================================

import urllib.request
import zipfile
import tensorflow as tf
import os
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, RMSprop

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def solution_A2():
    data_url_1 = 'https://dicodingacademy.blob.core.windows.net/picodiploma/Simulation/machine_learning/horse-or-human.zip'
    urllib.request.urlretrieve(data_url_1, 'horse-or-human.zip')
    local_file = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('dataset/horse-or-human')

    data_url_2 = 'https://dicodingacademy.blob.core.windows.net/picodiploma/Simulation/machine_learning/validation-horse-or-human.zip'
    urllib.request.urlretrieve(data_url_2, 'validation-horse-or-human.zip')
    local_file = 'validation-horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('dataset/validation-horse-or-human')
    zip_ref.close()

    TRAINING_DIR = 'dataset/horse-or-human'
    VAL_DIR = 'dataset/validation-horse-or-human'

    # YOUR CODE HERE
    train_datagen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=40,
        horizontal_flip=True,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,
        class_mode='binary',
        batch_size=64,
        target_size=(150, 150))

    test_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = test_datagen.flow_from_directory(VAL_DIR,
                                                            class_mode='binary',
                                                            batch_size=32,
                                                            target_size=(150, 150))

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy') > 0.83 and logs.get('val_accuracy') > 0.83):
                print("\n Accuracy is more than 83%, stopping...")
                self.model.stop_training = True

    customCallback = myCallback()

    model = tf.keras.models.Sequential([
        # YOUR CODE HERE, end with a Neuron Dense, activated by sigmoid
        Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[customCallback])

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_A2()
    model.save("model_A2.h5")
