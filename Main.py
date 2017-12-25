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

# Notes:
# *)
# figure out how the game will pass along positives and negatives values
# - Bad moves, good moves: jumping opponent, correct move, winning
# - Should winning even be a concern? it should all be about points for jumping w/
# - punishment of being jumped
# *)
# need some mapping of checkers actions to number of possible board position to new board position
# *)
# able to load in a model or start a new one?



checkers_game = "passed in"
print('Game Loaded')

# Game environment
checkers_actions = 8*8*(8*8)
# Provides the game any possible combination of moves on the checkers board


# ----------------------------------
# Create network for player 1 and player 2
# ----------------------------------
# qvalue_NN( checkers_actions ):
p1_model = qvalue_nn(checkers_actions)
p2_model = qvalue_nn(checkers_actions)

# ----------------------------------
# Q-value Parameters
# ----------------------------------
epsilon = 0.7          # Prob of doing a random move
gamma = 0.9            # Discounted future reward. How much we care about steps further in time

# ----------------------------------
# Q-value NN update Parameters
# ----------------------------------
observe_time = 150           # Number of time-steps we will be acting on the game and observing results
batch_size = 50              # Learning mini-batch size

# This is function will be where most of the game takes place
print("Entering the game:")
action_collect(p1_model, p2_model, epsilon, gamma, checkers_actions, observe_time, batch_size)

# How do I properly call for an action to occur and then begin to store each action for each player


# ----------------------------------
# Train model
# Learning from the observed actions -- Retraining of NN / Q-values
# ----------------------------------
# train_qvalue_nn(stored_actions, batch_size, model, gamma, checkers_actions, state)

#p1_model = train_qvalue_nn(p1_stored_actions, batch_size, p1_model, gamma, checkers_actions, p1_state)
#p2_model = train_qvalue_nn(p2_stored_actions, batch_size, p2_model, gamma, checkers_actions, p2_state)

#print('Learning Finished')

print("Thanks for watching 2 AIs play!")

