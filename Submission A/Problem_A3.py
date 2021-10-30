# ======================================================================================================
# PROBLEM A3
#
# Build a classifier for the Human or Horse Dataset with Transfer Learning.
# The test will expect it to classify binary classes.
# Note that all the layers in the pre-trained model are non-trainable.
# Do not use lambda layers in your model.
#
# The horse-or-human dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
# Inception_v3, pre-trained model used in this problem is developed by Google.
#
# Desired accuracy and validation_accuracy > 93%.
# -----------------------------------
# Muhammad Ilham Adhim
# =======================================================================================================

import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow.keras.applications.inception_v3 import InceptionV3


def solution_A3():
    inceptionv3 = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    urllib.request.urlretrieve(
        inceptionv3, 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
    local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

    # YOUR CODE HERE
    pre_trained_model = InceptionV3(input_shape=(
        150, 150, 3), include_top=False, weights=None)

    pre_trained_model.load_weights(local_weights_file)

    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_layer = pre_trained_model.get_layer('mixed7')
    last_output = last_layer.output

    data_url_1 = 'https://github.com/dicodingacademy/assets/releases/download/release-horse-or-human/horse-or-human.zip'
    urllib.request.urlretrieve(data_url_1, 'horse-or-human.zip')
    local_file = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('dataset/horse-or-human')

    data_url_2 = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/validation-horse-or-human.zip'
    urllib.request.urlretrieve(data_url_2, 'validation-horse-or-human.zip')
    local_file = 'validation-horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('dataset/validation-horse-or-human')
    zip_ref.close()

    x = Flatten()(last_output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(1, activation='sigmoid')(x)

    # model pre-trained
    model = Model(pre_trained_model.input, x)

    # model compile
    model.compile(optimizer=RMSprop(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['acc'])

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
            if(logs.get('acc') > 0.93 and logs.get('val_acc') > 0.93):
                print("\n Accuracy is more than 93%, stopping...")
                self.model.stop_training = True

    customCallback = myCallback()

    # train model
    model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=[customCallback]
    )
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_A3()
    model.save("model_A3.h5")
