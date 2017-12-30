import numpy as np
import random
from collections import deque            # For storing moves
from keras.models import Sequential     # One Layer after the other
from keras.layers import Dense, Flatten          # Dense layers are fully connected layers
import checkers
# ----------------------------------
# Collect Actions
# ----------------------------------
def action_decision(action):
    # will also return a score at some point
    return divmod(action, checkers.BOARD_SIZE)

