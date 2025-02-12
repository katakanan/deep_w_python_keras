import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
print(tf.__version__)
print(tf.keras.__version__)
# from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    num_sample_per_class = 1000
    negive_sample = np.random.multivariate_normal(mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=num_sample_per_class)
    positive_sample = np.random.multivariate_normal(mean=[3, 0], cov=[[1, 0.5], [0.5, 1]], size=num_sample_per_class)
    inputs = np.vstack((negive_sample, positive_sample)).astype(np.float32)
    targets = np.vstack((np.zeros((num_sample_per_class, 1), dtype="float32"), np.ones((num_sample_per_class, 1), dtype="float32")))

    inputs_dim = 2
    output_dim = 1
    W = tf.Variable(initial_value=tf.random.uniform(shape=(inputs_dim, output_dim)))
    b = tf.Variable(initial_value=tf.zeros(shape=(output_dim, )))

    def model(inputs):
        return tf.matmul(inputs, W) + b

    def square_loss(targets, predictions):
        per_sample_losses = tf.square(targets - predictions)
        return tf.reduce_mean(per_sample_losses)
    
    learning_rate = 0.1
    def training_step(inputs, targets):
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = square_loss(targets,predictions)

        grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
        W.assign_sub(grad_loss_wrt_W * learning_rate)
        b.assign_sub(grad_loss_wrt_b * learning_rate)
        return loss

    for step in range(40):
        loss = training_step(inputs, targets)
        print(f"Loss at step {step}: {loss:.4f}")

    predictions = model(inputs)

    x = np.linspace(-1, 4, 100)
    y = -W[0] / W[1] * x + (0.5 - b) / W[1]
    plt.plot(x, y, "-r")
    plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
    plt.show()