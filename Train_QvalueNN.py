from keras.models import Sequential     # One Layer after the other
from keras.layers import Dense, Flatten          # Dense layers are fully connected layers
import numpy as np
import random
from collections import deque            # For storing moves


def train_qvalue_nn(stored_actions, batch_size, model, gamma, checkers_actions, state):
    batch_update = random.sample(stored_actions, batch_size)
    inputs_shape = (batch_size,) + state.shape[1:]
    inputs = np.zeros(inputs_shape)
    targets = np.zeros((batch_size, checkers_actions))
    for i in range(0, batch_size):
        state = batch_update[i][0]
        action = batch_update[i][1]
        reward = batch_update[i][2]
        state_new = batch_update[i][3]
        done = batch_update[i][4]
        # Build Bellman equation for the Q function
        inputs[i:i + 1] = np.expand_dims(state, axis=0)
        targets[i] = model.predict(state)
        Q_sa = model.predict(state_new)
        if done:
            targets[i, action] = reward
        else:
            targets[i, action] = reward + gamma * np.max(Q_sa)
        # Train network to output the Q function
        model.train_on_batch(inputs, targets)
    return model
