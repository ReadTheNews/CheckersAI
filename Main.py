# import tensorflow as tf
# import keras
from keras.models import Sequential     # One Layer after the other
from keras.layers import Dense, Flatten          # Dense layers are fully connected layers
import numpy as np
from collections import deque            # For storing moves

import sys
sys.version

# Custom Functions
from Qvalue_NN import qvalue_nn
from Action_collect import action_collect
from Train_QvalueNN import train_qvalue_nn
from Action_decision import action_decision

import checkers

## TO DO LIST
# - Work on rewards and penalties
# - Work on maximum number of aloud moves to reset board an penalize

print('Game Loaded')

# Game environment
checkers_actions = checkers.BOARD_SIZE*checkers.BOARD_SIZE
# Provides the game any possible combination of moves on the checkers board

# ----------------------------------
# Create network for player 1 and player 2
# ----------------------------------
# qvalue_NN( checkers_actions ):
print('Would you like to load in a model or start over?')
p1_model = qvalue_nn(checkers_actions)
p2_model = qvalue_nn(checkers_actions)
load_model = ""
while load_model !='y' and load_model != 'n':
    load_model = input("Load? 'y'/'n': ")
if load_model == 'y':
    p1_model.load_weights('White_model_weights.h5')
    p2_model.load_weights('Black_model_weights.h5')
    print("Models loaded")
else:
    print("New models created")

# ----------------------------------
# Q-value Parameters
# ----------------------------------
epsilon = 0.7          # Prob of doing a random move
gamma = 0.9            # Discounted future reward. How much we care about steps further in time

# ----------------------------------
# Q-value NN update Parameters
# ----------------------------------
observe_time = 2000           # Number of time-steps we will be acting on the game and observing results
batch_size = 1000              # Learning mini-batch size

# This is function will be where most of the game takes place
print("Entering the game:")
action_collect(p1_model, p2_model, epsilon, gamma, checkers_actions, observe_time, batch_size)



print("Thanks for watching 2 AIs play!")

