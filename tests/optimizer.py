import tensorflow as tf
import numpy as np

X = tf.Variable([0.2, -0.3])

optimizer = tf.keras.optimizers.Adam(learning_rate=1)
for _ in range(10):
    optimizer.minimize(lambda: sum(tf.square(X)), [X])
    print(X)