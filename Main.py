# import tensorflow as tf
# import keras
from keras.models import Sequential     # One Layer after the other
from keras.layers import Dense, Flatten          # Dense layers are fully connected layers
import numpy as np
import random
from collections import deque            # For storing moves
import Qvalue_NN

# Notes:
# 1)
# figure out how the game will pass along positives and negatives values
# - Bad moves, good moves: jumping opponent, correct move, winning
# - Should winning even be a concern? it should all be about points for jumping w/
# - punishment of being jumped
# 2)
# need to make some type of counter that stores "x" number of states for training of the NN
# Then we will enter the loop to train from the sampled moves
# post training - dump the previous moves to collect and retrain.
# 3)
# need some mapping of checkers actions to number of possible board position to new board position
# 4)
# able to load in a model or start a new one?
# 5) program should be a function
# be able to call so it could represent 2 players with 1 code


checkers_game = "passed in"
print('Game Loaded')

# Game environment
checkers_actions = 8*8*(8)   # This should be total number of possible actions to choose from
# 8 possible moves - move [ up-left, up-right, down-left, down-right] * (can't jump, can jump)
# 8*8 = piece selection (maybe reduce the number of choices? Since some will be null

# ----------------------------------
# Create network for player 1 and player 2
# ----------------------------------
p1_model = Qvalue_NN(checkers_actions)
p2_model = Qvalue_NN(checkers_actions)

# ----------------------------------
# Q-value Parameters
# ----------------------------------
epsilon = 0.7          # Prob of doing a random move
gamma = 0.9            # Discounted future reward. How much we care about steps further in time

# ----------------------------------
# Q-value NN update Parameters
# ----------------------------------
stored_actions = deque()      # Register where the actions will be stored
observe_time = 1000           # Number of time-steps we will be acting on the game and observing results
batch_size = 300              # Learning mini-batch size


# ----------------------------------
# Initialization for model
# ----------------------------------
observation = ""       # Observation is passed in
obs = np.expand_dims(observation, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs
state = np.stack((obs, obs), axis=1) # what exactly is happening here?
done = False
# ----------------------------------
# Collect Actions
# ----------------------------------


print('Observing Finished')
# ----------------------------------
# Train model
# Learning from the observed actions -- Retraining of NN / Q-values
# ----------------------------------
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
print('Learning Finished')


