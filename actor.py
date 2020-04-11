import tensorflow as tf

class Actor:
    
    def __init__(self, input_dims, hidden_layers, num_of_actions, bounds, critic, activation='relu', tau=0.001, alpha=0.0001, orig_layers=[]):

        self.critic = critic
        self.tau = tau

        inp = tf.keras.layers.Input(shape=(input_dims, ))
        cur_layer = inp
        for layer in orig_layers:
            cur_layer = layer(cur_layer)

        first = len(orig_layers) == 0

        for dim in hidden_layers:
            if first:
                cur_layer = tf.keras.layers.Dense(dim, input_shape=(input_dims, ), activation=activation)(cur_layer)
            else:
                cur_layer = tf.keras.layers.Dense(dim, activation=activation)(cur_layer)
            cur_layer = tf.keras.layers.BatchNormalization()(cur_layer)

        output_layers = []

        for i in range(num_of_actions):
            output_layer = tf.keras.layers.Dense(1, activation='sigmoid' if bounds[i][0] == 0 else 'tanh')
            output_layer = tf.keras.layer.Lambda(lambda x: x * bounds[i][1])
            output_layers.append(output_layer)

        out = tf.keras.layers.Concatenate()(output_layers)
        
        self.model = tf.keras.models.Model(input=inp, output=out)
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)

    # TODO: test and fix any errors
    def train(self, states):
        with tf.GradientTape() as tape:
            loss = -tf.keras.backend.mean(self.critic.predict(states, self.predict(states)))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def update_target_model(self):
        model_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def get_model(self):
        return self.model

    def get_target_mode(self):
        return self.target_model

    def load_model(self, model_file):
        self.model.load_weights(model_file)
        self.target_model.load_weights(model_file)

    def save_model(self, model_file):
        self.model.save_weights(model_file)