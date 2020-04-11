from critic import Critic
from actor import Actor
from replay_buffer import ReplayBuffer

class Agent:

    def __init__(self, alpha, gamma, num_of_actions, tau, batch_size, input_dims, ADD_MORE_HERE, mem_size=1000000):
        self.memory = ReplayBuffer(mem_size, input_dims, num_of_actions, discrete=True)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)