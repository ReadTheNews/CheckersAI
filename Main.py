# import tensorflow as tf
# import keras
from keras.models import Sequential     # One Layer after the other
from keras.layers import Dense          # Dense layers are fully connected layers
import numpy as np
import random
from collections import deque            # For storing moves
# import sys
# sys.version
# print("Hello World")

# Notes:
# 1)
# figure out how the game will pass along positives and negatives values
# - Bad moves, good moves: jumping opponent, correct move, winning
# 2)
# need to make some type of counter that stores "x" number of states for training of the NN
# Then we will enter the loop to train from the sampled moves
# post training - dump the previous moves to collect and retrain.
# 3)
# need some mapping of checkers actions to number of possible board position to new board position
# 4)
# able to load in a model or start a new one?


# Game environment
checkers_actions = 8*8*(8)   # This should be total number of possible actions to choose from
# 8 possible moves - move [ up-left, up-right, down-left, down-right] * (can't jump, can jump)
# 8*8 = piece selection (maybe reduce the number of choices? Since some will be null

# Create network. Input is two consecutive game states, output is Q-values of the possible moves.
model = Sequential()
model.add(Dense(64, input_dim=2, kernel_initializer='uniform', activation='relu'))
model.add(Dense(40, kernel_initializer='uniform', activation='relu'))
model.add(Dense(20, kernel_initializer='uniform', activation='relu'))
# Next layer - same number of outputs as possible actions
model.add(Dense(env_actions, kernel_initializer='uniform', activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Q-value Parameters
epsilon = 0.7          # Prob of doing a random move
gamma = 0.9            # Discounted future reward. How much we care about steps further in time

# NN update Parameters
stored_actions = deque()      # Register where the actions will be stored
observe_time = 1000           # Number of time-steps we will be acting on the game and observing results
batch_size = 300              # Learning mini-batch size

# Initialization for model
# -- What is going on in here?
observation = ""        # previously env.reset()          # Game begins
obs = np.expand_dims(observation, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs
state = np.stack((obs, obs), axis=1)
done = False

# The Q-value formula + Collect Actions
for t in range(observe_time):
    if np.random.rand() <= epsilon:
        action = np.random.randint(0, checkers_actions, size=1)[0]
    else:
        Q = model.predict(state)          # Q-values predictions
        action = np.argmax(Q)             # Move with highest Q-value is the chosen one
    observation_new, reward, done, info = env.step(action)     # See state of the game, reward... after performing the action
    obs_new = np.expand_dims(observation_new, axis=0)          # (Formatting issues)
    state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)     # Update the input with the new state of the game
    D.append((state, action, reward, state_new, done))         # 'Remember' action and consequence
    state = state_new         # Update state
    if done:
        ## Need to learn what is happening inside the check
        env.reset()           # Restart game if it's finished
        obs = np.expand_dims(observation, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs
        state = np.stack((obs, obs), axis=1)
print('Observing Finished')



# Learning from the observed actions -- Retraining of NN / Q-values

batch_update = random.sample(stored_actions, batch_size)




