import numpy as np
import random

from brain import Brain
from memory import Memory
from sklearn.preprocessing import StandardScaler
MAX_EPSILON = 1.0
MIN_EPSILON = 0.01

MIN_BETA = 0.4
MAX_BETA = 1.0
#  def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim)

class Agent:
    
    epsilon = MAX_EPSILON
    beta = MIN_BETA
    def __init__(self,bee_index, config):
        self.state_size = config['num_states']
        self.action_size = config['num_actions']
        self.bee_index = bee_index
        self.learning_rate = config['learning_rate']
        self.gamma = 0.95
        self.brain = Brain(config['num_layers'], 
                    config['width_layers'], 
                    config['batch_size'], 
                    config['learning_rate'], 
                    input_dim=config['num_states'], 
                    output_dim=config['num_actions'])

        self.memory= Memory( config['memory_size_max'], config['memory_size_min'] )

        self.max_exploration_step = config['maximum_exploration']
        self.batch_size = config['batch_size']
        self.step = 0

    def greedy_actor(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.brain.predict_one_sample(state))
    
    def observe(self, sample):
        self.memory.add_sample(sample)
    
    def decay_epsilon(self):
        # slowly decrease Epsilon based on our experience
        self.step += 1
        if self.step < self.max_exploration_step:
            self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * (self.max_exploration_step - self.step)/self.max_exploration_step
            self.beta = MAX_BETA + (MIN_BETA - MAX_BETA) * (self.max_exploration_step - self.step)/self.max_exploration_step
        else:
            self.epsilon = MIN_EPSILON
    
    def replay(self):
        batch = self.memory.get_samples(self.batch_size)
        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            q_s_a = self.brain.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self.brain.predict_batch(next_states)  # predict Q(next_state), for every sample

            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))
            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1][self.bee_index], b[2], b[3] 
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self.gamma * np.amax(q_s_a_d[i])
                x[i] = state
                y[i] = current_q  
            scaler = StandardScaler()
            x = scaler.fit_transform(x)
            y = scaler.fit_transform(y)
            self.brain.train_batch(x, y)
