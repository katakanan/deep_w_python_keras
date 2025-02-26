import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

if __name__ == "__main__":
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # model.summary()
    (train_images, train_label), (test_images, test_label) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype("float32") / 255
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype("float32") / 255
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_images, train_label, epochs=5, batch_size=64)
    test_loss, test_acc = model.evaluate(test_images, test_label)
    print(f"Test accuracy: {test_acc:3f}")

