import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt

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

    model = keras.Sequential([
                                layers.Dense(16, activation="relu"),
                                layers.Dense(16, activation="relu"),
                                layers.Dense(1, activation="sigmoid"),
                            ])
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
    history_dict = history.history
    print(history_dict.keys())
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("Training and Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()