import numpy as np
import random
from collections import deque            # For storing moves
from keras.models import Sequential     # One Layer after the other
from keras.layers import Dense, Flatten          # Dense layers are fully connected layers

from Action_decision import action_decision

# ----------------------------------
# Collect Actions
# ----------------------------------
def action_collect(p1_model, p1_state, p2_model, p2_state, epsilon, checkers_actions, observe_time, \
                   p1_stored_actions, p2_stored_actions, iterations):
    checkers_game = "" # need to import checkers game path???
    done = False

    while iterations < observe_time:
        player_turn = checkers_game.players_turn

        if player_turn == 1:
            model = p1_model
            state = p1_state
        elif player_turn == 2:
            model = p2_model
            state = p2_state
        else:
            print("error")

        if np.random.rand() <= epsilon:
            action = np.random.randint(0, checkers_actions, size=1)[0]
        else:
            q_value = model.predict(state)                        # Q-values predictions
            action = np.argmax(q_value)                           # Move with highest Q-value is the chosen one

        observation_new, reward = action_decision(action, iterations, p1_model, p1_state, p2_model, p2_state, \
                                                  epsilon, checkers_actions, observe_time, \
                                             p1_stored_actions, p2_stored_actions, iterations)
                                                 # See state of the game, reward... after performing the action
        obs_new = np.expand_dims(observation_new, axis=0)          # (Formatting issues)
        # make sure I under stand the below function
        state_new = np.append(np.expand_dims(obs_new, axis=0), p1_state[:, :1, :], axis=1)     # Update the input with the new state of the game


        if player_turn == 1:

            p1_stored_actions.append((p1_state, action, reward, state_new, done))  # 'Remember' action and consequence
            p1_state = state_new         # Update state
        if done:
                    # force action collect to end.
                    # also need to  update p2's state and actions
                    return ""

                    # Will put into place later, not now
                    # checkers_game.reset()
                    # obs = np.expand_dims(observation, axis=0)
                        # (Formatting issues) Making the observation the first element of a batch of inputs
                    # state = np.stack((obs, obs), axis=1)  # what exactly is happening here?

    return p1_stored_actions, p2_stored_actions
