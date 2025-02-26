import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# from tensorflow import keras
import keras
from keras import layers

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
    
    model.summary()