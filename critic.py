import tensorflow as tf

class Critic:

    def __init__(self, state_size, num_of_actions, state_hidden_layers, action_hidden_layers, merge_dim, merged_hidden_layers, activation='relu', tau=0.001, alpha=0.0001, orig_layers=[]):
        
        self.tau = tau

        state_inp = tf.keras.layers.Input(shape=(state_size, ))
        action_inp = tf.keras.layers.Input(shape=(num_of_actions, ))

        cur_state_layer = state_inp
        for dim in state_hidden_layers + [merge_dim]:
            cur_state_layer = tf.keras.layers.Dense(dim, activation=activation)(cur_state_layer)
            cur_state_layer = tf.keras.layers.BatchNormalization()(cur_state_layer)

        cur_action_layer = action_inp
        for dim in action_hidden_layers + [merge_dim]:
            cur_action_layer = tf.keras.layers.Dense(dim, activation=activation)(cur_action_layer)
            cur_action_layer = tf.keras.layers.BatchNormalization()(cur_action_layer)

        cur_layer = tf.keras.layers.Add()([cur_state_layer, cur_action_layer])
        for dim in merged_hidden_layers:
            cur_layer = tf.keras.layers.Dense(dim, activation=activation)
            cur_layer = tf.keras.layers.BatchNormalization()(cur_layer)
        out = tf.keras.layers.Dense(num_of_actions)
        
        self.model = tf.keras.models.Model(input=[state_inp, action_inp], output=out)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha), loss='mse')
        
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def train(self, inp, target):
        self.model.fit(inp, target)

    def update_target_model(self):
        model_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def get_model(self):
        return self.model

    def get_target_model(self):
        return self.target_model