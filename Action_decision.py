import numpy as np
import random
from collections import deque            # For storing moves
from keras.models import Sequential     # One Layer after the other
from keras.layers import Dense, Flatten          # Dense layers are fully connected layers

# ----------------------------------
# Collect Actions
# ----------------------------------
def action_decision(action, iterations, p1_model, p1_state, p2_model, p2_state, \
                         epsilon, checkers_actions, observe_time, p1_stored_actions, p2_stored_actions):
    checkers_game = ""  # need to import checkers game path???
    if iterations < observe_time:


        action_collect