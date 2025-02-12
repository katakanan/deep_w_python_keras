import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
print(tf.__version__)
print(tf.keras.__version__)
# from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt

num_sample_per_class = 1000
negive_sample = np.random.multivariate_normal(mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=num_sample_per_class)
positive_sample = np.random.multivariate_normal(mean=[3, 0], cov=[[1, 0.5], [0.5, 1]], size=num_sample_per_class)
inputs = np.vstack((negive_sample, positive_sample)).astype(np.float32)
targets = np.vstack((np.zeros((num_sample_per_class, 1), dtype="float32"), np.ones((num_sample_per_class, 1), dtype="float32")))

plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.show()