import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.datasets import imdb
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.

    return results

if __name__ == "__main__":
    (train_data, train_label), (test_data, test_label) = imdb.load_data(num_words = 10000)

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    y_train  = np.asarray(train_label).astype("float32")
    y_test = np.asarray(test_label).astype("float32")