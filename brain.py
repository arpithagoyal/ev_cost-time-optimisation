import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models, regularizers


class Brain:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width)
        self._target_model=self._build_model(num_layers,width)


    def _build_model(self, num_layers, width):
        """
        Build and compile a fully connected deep neural network
        """
        inputs = keras.Input(shape=(self._input_dim,))
        x = layers.Dense(width, activation='relu')(inputs)
        for _ in range(num_layers):
            x = layers.Dense(width, activation='relu')(x)
        outputs = layers.Dense(self._output_dim, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')
        model.compile(loss=losses.MeanSquaredError, optimizer=Adam(learning_rate=self._learning_rate))
        return model
    
    def predict_one(self, state,target=False):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        if target:
            return self._target_model.predict(state)
        else:
            return self._model.predict(state)


    def predict_batch(self, states, target=False):
        """
        Predict the action values from a batch of states
        """
        if(target):
            return self._target_model.predict(states)
        else:
            return self._model.predict(states)


    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        self._model.fit(states, q_sa, epochs=1, verbose=1)


    def update_target_model(self):
        self._target_model.set_weights(self.model.get_weights())

    def save_model(self):
        self._model.save(self.weight_backup)


    @property
    def input_dim(self):
        return self._input_dim


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size