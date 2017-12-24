import numpy as np
import random
from collections import deque            # For storing moves
from keras.models import Sequential     # One Layer after the other
from keras.layers import Dense, Flatten          # Dense layers are fully connected layers

from Action_decision import action_decision # Decision mapping function
import checkers as game

# ----------------------------------
# Collect Actions
# ----------------------------------
def action_collect(p1_model, p2_model, epsilon, gamma, checkers_actions, observe_time, batch_size):
    # ----------------------------------
    # Initialization for model -- First game in training?
    # ----------------------------------

    checkers_game = game.board_new()  # need to import checkers game path???
    #print(checkers_game)

    obs = np.expand_dims(checkers_game, axis=0)
    #print(obs)
    p1_state = np.stack((obs, obs), axis=1)
    p2_state = np.stack((obs, obs), axis=1)
    #print(p1_state)

    # ----------------------------------
    # Collect Actions
    # ----------------------------------
    # action_collect(model, state, epsilon, checkers_actions, observe_time )
    p1_stored_actions = deque()  # Register where the actions will be stored
    p2_stored_actions = deque()  # Register where the actions will be stored

    #done = False

    player_turn = game.Player.WHITE
    last_player = ""
    # print(str(player_turn))
    usr_input = ""

    # Inside this function -- the game should run "Endlessly"

    while usr_input != "Escape":
    # while iterations < observe_time:
        print(str(game.Player.))
        if str(player_turn) == last_player:
            print("\nBad Move!")


        if str(player_turn) == "Player.WHITE":
            model = p1_model
            state = p1_state
            print("Player is White")
        elif str(player_turn) == "Player.BLACK":
            model = p2_model
            state = p2_state
            print("Player is Black")
        else:
            print("error")

        if np.random.rand() <= epsilon:
            action = np.random.randint(0, checkers_actions, size=1)[0]
            print("Random")
        else:
            q_value = model.predict(state)                        # Q-values predictions
            action = np.argmax(q_value)                           # Move with highest Q-value is the chosen one
            print("Predicted")

        action_mapping = action_decision(action)
        print(action_mapping)
        moves = action_mapping.split(',')
        print(str(int(moves[0])) + "-" + str(int(moves[1])) + "-")
        game.move(checkers_game, player_turn, int(moves[0]), int(moves[1]))


        usr_input = input("Please type 'Escape' if you choose to end: ")
        print("You entered " + str(usr_input))
        last_player = str(player_turn)


        #obs_new = np.expand_dims(observation_new, axis=0)          # (Formatting issues)
        # make sure I under stand the below function
        #state_new = np.append(np.expand_dims(obs_new, axis=0), p1_state[:, :1, :], axis=1)     # Update the input with the new state of the game


        #if player_turn == 1:

        #p1_stored_actions.append((p1_state, action, reward, state_new, done))  # 'Remember' action and consequence
        #p1_state = state_new         # Update state
        #if done:
                    # force action collect to end.
                    # also need to  update p2's state and actions
        #           return ""

                    # Will put into place later, not now
                    # checkers_game.reset()
                    # obs = np.expand_dims(observation, axis=0)
                        # (Formatting issues) Making the observation the first element of a batch of inputs
                    # state = np.stack((obs, obs), axis=1)  # what exactly is happening here?

    return "Hellow World"
    #return p1_stored_actions, p2_stored_actions
