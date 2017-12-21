# import tensorflow as tf
# import keras
from keras.models import Sequential     # One Layer after the other
from keras.layers import Dense          # Dense layers are fully connected layers
import numpy as np

# import collections import deque -- storing moves????

import sys
sys.version
print("Hello World")



# Create network. Input is two consecutive game states, output is Q-values of the possible moves.
model = Sequential()
model.add(Dense(64, input_dim=2, kernel_initializer='uniform', activation='relu'))
model.add(Dense(40, kernel_initializer='uniform', activation='relu'))
model.add(Dense(20, kernel_initializer='uniform', activation='relu'))
model.add(Dense(env.action_space.n, kernel_initializer='uniform', activation='linear')) # same number of outputs as possible actions
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Parameters
observ_t = 6000        # Number of time-steps we will be acting on the game and observing results
epsilon = 0.7          # Prob of doing a random move
gamma = 0.9            # Discounted future reward. How much we care about steps further in time
mb_size = 300          # Learning mini-batch size
