import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.

    return results

def to_one_hot(labels, dimensions=46):
    results = np.zeros((len(labels), dimensions))
    for i, label in enumerate(labels):
        results[i, label] = 1.

    return results

if __name__ == "__main__":
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000)
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    y_train  = to_one_hot(train_labels)
    y_test = to_one_hot(test_labels)
    # y_train  = to_categorical(train_labels)
    # y_test = to_categorical(test_labels)

    model = keras.Sequential([
                                layers.Dense(64, activation="relu"),
                                layers.Dense(64, activation="relu"),
                                layers.Dense(46, activation="softmax"),
                            ])
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    # x_val = x_train[:1000]
    # partial_x_train = x_train[1000:]
    # y_val = y_train[:1000]
    # partial_y_train = y_train[1000:]
    # history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
    model.fit(x_train, y_train, epochs=9, batch_size=512)
    results = model.evaluate(x_test, y_test)
    print(results)

    # history_dict = history.history
    # print(history_dict.keys())
    # loss_values = history_dict["loss"]
    # val_loss_values = history_dict["val_loss"]
    # epochs = range(1, len(loss_values) + 1)
    # plt.figure()
    # plt.plot(epochs, loss_values, "bo", label="Training loss")
    # plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    # plt.title("Training and Validation loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.legend()

    # plt.figure()
    # acc = history_dict["accuracy"]
    # val_acc = history_dict["val_accuracy"]
    # plt.plot(epochs, acc, "bo", label="Training acc")
    # plt.plot(epochs, val_acc, "b", label="Validation acc")
    # plt.title("Training and Validation acc")
    # plt.xlabel("Epochs")
    # plt.ylabel("Accucary")
    # plt.legend()
    # plt.show()