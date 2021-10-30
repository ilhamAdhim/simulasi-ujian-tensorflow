# ==========================================================================================================
# PROBLEM A4
#
# Build and train a binary classifier for the IMDB review dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in http://ai.stanford.edu/~amaas/data/sentiment/
#
# Desired accuracy and validation_accuracy > 83%
# -----------------------------------
# Muhammad Ilham Adhim
# ===========================================================================================================

import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing.text import text_to_word_sequence
import tensorflow_datasets as tfds
import numpy as np

from keras.models import Sequential

from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Dense, Embedding, Dropout, Flatten
from keras.preprocessing.sequence import pad_sequences


def solution_A4():
    imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
    train_data, test_data = imdb['train'], imdb['test']
    # YOUR CODE HERE
    # training sentences dan labels
    training_sentences = []
    training_labels = []

    # testing sentences dan testing labels
    testing_sentences = []
    testing_labels = []

    for s, l in train_data:
        training_sentences.append(s.numpy().decode('utf8'))
        training_labels.append(l.numpy())

    for s, l in test_data:
        testing_sentences.append(s.numpy().decode('utf8'))
        testing_labels.append(l.numpy())

    # YOUR CODE HERE
    training_labels_updated = np.array(training_labels)
    testing_labels_updated = np.array(testing_labels)

    vocab_size = 10000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    oov_tok = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)

    seq = tokenizer.texts_to_sequences(training_sentences)
    test_seq = tokenizer.texts_to_sequences(testing_sentences)

    padded = pad_sequences(seq, maxlen=max_length, truncating=trunc_type)
    testing_padded = pad_sequences(test_seq, maxlen=max_length)

    # modelling
    model = Sequential([
        # YOUR CODE HERE. Do not change the last layer.
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        Bidirectional(tf.keras.layers.GRU(16)),
        Flatten(),
        Dropout(0.2),
        Dense(6, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    class handleCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy') >= 0.83 and logs.get('val_accuracy') >= 0.83):
                print("\nTrain Accuracy and Val_Accuracy is more than 83%!")
                print("\nTraining stopped...")
                self.model.stop_training = True

    callbacks = handleCallback()

    # compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # training model
    model.fit(
        padded,
        training_labels_updated,
        epochs=10,
        validation_data=(testing_padded, testing_labels_updated),
        callbacks=[callbacks])

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_A4()
    model.save("model_A4.h5")
