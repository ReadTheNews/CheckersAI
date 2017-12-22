import numpy as np
import random
from collections import deque            # For storing moves
from keras.models import Sequential     # One Layer after the other
from keras.layers import Dense, Flatten          # Dense layers are fully connected layers

# ----------------------------------
# Collect Actions
# ----------------------------------
def action_collect(model, state, epsilon, checkers_actions, observe_time ):
    checkers_game = "" # need to import checkers game path???
    for t in range(observe_time):
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, checkers_actions, size=1)[0]
        else:
            Q = model.predict(state)                        # Q-values predictions
            action = np.argmax(Q)                           # Move with highest Q-value is the chosen one
        observation_new, reward = checkers_game(action)     # See state of the game, reward... after performing the action
        obs_new = np.expand_dims(observation_new, axis=0)          # (Formatting issues)
        # make sure I under stand the below function
        state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)     # Update the input with the new state of the game
        stored_actions.append((state, action, reward, state_new, done))  # 'Remember' action and consequence
        state = state_new         # Update state
        if done:
                checkers_game.reset()
                obs = np.expand_dims(observation, axis=0)
                    # (Formatting issues) Making the observation the first element of a batch of inputs
                state = np.stack((obs, obs), axis=1)  # what exactly is happening here?
    return ""
