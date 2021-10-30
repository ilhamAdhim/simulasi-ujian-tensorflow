
# =============================================================================
# PROBLEM B2
#
# Build a classifier for the Fashion MNIST dataset.
# The test will expect it to classify 10 classes.
# The input shape should be 28x28 monochrome. Do not resize the data.
# Your input layer should accept (28, 28) as the input shape.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 83%
# ------------------------
# Muhammad Ilham Adhim
# =============================================================================

import urllib.request
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, RMSprop

from tensorflow.keras.layers import Flatten, Dense


def solution_B2():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images,
                                         test_labels) = fashion_mnist.load_data()

    training_images = training_images / 255.0
    test_images = test_images / 255.0

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy') > 0.83 and logs.get('val_accuracy') > 0.83):
                print("\n Accuracy is more than 83%, stopping...")
                self.model.stop_training = True

    custom_callback = myCallback()
    # YOUR CODE HERE
    model = Sequential([
        Flatten(),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(training_images, training_labels, epochs=5, validation_data=(
        test_images, test_labels), callbacks=[custom_callback])

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_B2()
    model.save("model_B2.h5")
