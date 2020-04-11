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
optimizer = tf.keras.optimizers.Adam()

inputs = tf.convert_to_tensor([
    [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]
])

outputs = tf.convert_to_tensor([
    [1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 1.0]
])

# loss_fn = lambda:tf.keras.losses.mse(model.predict(inputs), outputs)
# print(loss_fn())
# epoch = 10

# print(model.predict(inputs))
# for _ in range(epoch):
#     optimizer.minimize(loss_fn, model.trainable_variables)
# print(model.predict(inputs))

def train_step(inp, out):
    with tf.GradientTape() as tape:
        tape.watch(inp)
        tape.watch(out)
        predictions = model.predict(inp)
        tape.watch(predictions)
        loss = inp
    gradients = tape.gradient(loss, model.trainable_variables)
    print(len(gradients), len(model.trainable_variables))
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

epoch = 10

print(model.predict(inputs))

for _ in range(epoch):
    for inp, out in zip(inputs, outputs):
        train_step(inp, out)
print(model.predict(inputs))