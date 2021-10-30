# =====================================================================================================
# PROBLEM C4
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
#
# Do not use lambda layers in your model.
#
# Dataset used in this problem is built by Rishabh Misra (https://rishabhmisra.github.io/publications).
#
# Desired accuracy and validation_accuracy > 75%
# ------------------------
# Muhammad Ilham Adhim
# =======================================================================================================

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_C4():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sarcasm.json'

    urllib.request.urlretrieve(data_url, 'sarcasm.json')

    with open("/content/sarcasm.json", 'r') as f:
        datastore = json.loads("[" +
                               f.read().replace("}\n{", "},\n{") +
                               "]")

    sentences = []
    labels = []

    for item in datastore:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    # YOUR CODE HERE
    training_sentences = sentences[:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[:training_size]
    testing_labels = labels[training_size:]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)

    word_index = tokenizer.word_index

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(
        training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(
        testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    training_padded = np.array(training_padded)
    training_labels = np.array(training_labels)
    testing_padded = np.array(testing_padded)
    testing_labels = np.array(testing_labels)

    model = tf.keras.Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        GlobalAveragePooling1D(),
        Dense(24, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy') > 0.75 and logs.get('val_accuracy') > 0.75):
                print("\nAccuracy and Validation accuracy is more than 75%!")
                print("\nStopping training...")
                self.model.stop_training = True

    callbacks = CustomCallback()

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(training_padded,
                        training_labels,
                        epochs=20,
                        validation_data=(testing_padded, testing_labels),
                        verbose=1,
                        callbacks=[callbacks])

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_C4()
    model.save("model_C4.h5")
