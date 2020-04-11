import tensorflow as tf

input_dims = 2
activation = 'tanh'
hidden_layers = [4, 4]
outputs = 2

layers = []
first = True
for dim in hidden_layers:
    if first:
        layers.append(tf.keras.layers.Dense(dim, input_shape=(input_dims, ), activation=activation))
    else:
        layers.append(tf.keras.layers.Dense(dim, activation=activation))
layers.append(tf.keras.layers.Dense(outputs))
model = tf.keras.Sequential(layers)
# optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer='adam', loss='mse')

inputs = [
    [1, 0], [0, 1], [0, 0], [1, 1]
]

outputs = [
    [1, 1], [1, 1], [0, 0], [0, 1]
]

# model.fit(inputs, outputs, epochs=1000)



# print(model.predict(inputs))

print(type(model.trainable_weights))