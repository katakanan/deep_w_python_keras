import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import reuters
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000)
    print(len(train_data))
    print(len(test_data))
    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_review = " ".join([reverse_word_index.get(i - 3, "?") for i in train_data[0]])
    print(decoded_review)