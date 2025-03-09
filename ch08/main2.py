import os, shutil, pathlib
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

if __name__ == "__main__":
    script_dir = pathlib.Path(os.path.dirname(__file__))
    original_dir = script_dir / "../../PetImages"
    new_base_dir = script_dir / "cats_vs_dogs_small"

    train_dataset = image_dataset_from_directory(new_base_dir / "train", image_size=(180, 180), batch_size=32)
    validation_dataset = image_dataset_from_directory(new_base_dir / "validation", image_size=(180, 180), batch_size=32)
    test_dataset = image_dataset_from_directory(new_base_dir / "test", image_size=(180, 180), batch_size=32)

    # inputs = keras.Input(shape=(180, 180, 3))
    # x = layers.Rescaling(1./255)(inputs)
    # x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPooling2D(pool_size=2)(x)
    # x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPooling2D(pool_size=2)(x)
    # x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPooling2D(pool_size=2)(x)
    # x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPooling2D(pool_size=2)(x)
    # x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    # x = layers.Flatten()(x)
    # outputs = layers.Dense(1, activation="sigmoid")(x)
    # model = keras.Model(inputs=inputs, outputs=outputs)

    # # model.summary()
    # model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    # callbacks = [
    #                 keras.callbacks.ModelCheckpoint(filepath="convnet_from_scratch.keras", save_best_only=True, monitor="val_loss")
    #             ]
    
    # history = model.fit(train_dataset, epochs=30, validation_data=validation_dataset, callbacks=callbacks)

    # accuracy = history.history["accuracy"]
    # val_accuracy = history.history["val_accuracy"]
    # loss = history.history["loss"]
    # val_loss = history.history["val_loss"]
    # epochs = range(1, len(accuracy)+1)
    # plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    # plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    # plt.title("Training and validation accuracy")
    # plt.legend()
    # plt.figure()
    # plt.plot(epochs, loss, "bo", label="Training loss")
    # plt.plot(epochs, val_loss, "b", label="Validation loss")
    # plt.title("Training and validation loss")
    # plt.legend()
    # plt.show()

    test_model = keras.models.load_model("convnet_from_scratch.keras")
    test_loss, test_acc = test_model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc:.3f}")
