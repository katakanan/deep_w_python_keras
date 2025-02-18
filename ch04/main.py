import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.datasets import imdb

if __name__ == "__main__":
    (train_data, train_label), (test_data, test_label) = imdb.load_data(num_words = 10000)

    # print(train_data[0])
    # print(train_label[0])

    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_review = " ".join([reverse_word_index.get(i - 3, "?") for i in train_data[0]])
    print(decoded_review)