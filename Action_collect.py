import numpy as np
import random
from collections import deque            # For storing moves
from keras.models import Sequential     # One Layer after the other
from keras.layers import Dense, Flatten          # Dense layers are fully connected layers


from Action_decision import action_decision  # Decision mapping function
import checkers
from Train_QvalueNN import train_qvalue_nn
from time import sleep



# ----------------------------------
# Collect Actions
# ----------------------------------
def action_collect(p1_model, p2_model, epsilon, gamma, checkers_actions, observe_time, batch_size):
    # ----------------------------------
    # Initialization for model -- First game in training?
    # ----------------------------------

    checkers_game = checkers.Game()  # need to import checkers game path???
    #print(checkers_game)

    obs = np.expand_dims(np.asarray(checkers_game.board), axis=0)
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

    player_turn = checkers_game.turn
    last_player = ""
    last_board_state = np.asarray((64,))
    # print(str(player_turn))
    usr_input = ""
    new_game = True
    first_Black_Turn = True
    # Inside this function -- the game should run "Endlessly"

    while usr_input != "Escape":
        out_of_time = False
        while len(p1_stored_actions) < observe_time and len(p2_stored_actions) < observe_time:
            # This should print out the "current" player and will check if the player is different than the previous state
            player_turn = checkers_game.turn
            # print(str(game.Player.))
            # if str(player_turn) == last_player:
                # print("\nBad Move!")
            # else:
                # print("\nNext Players Turn")


            if str(player_turn) == "Player.WHITE":
                model = p1_model
                state = p1_state
                #print("Player is White")
            elif str(player_turn) == "Player.BLACK":
                model = p2_model
                state = p2_state
                #print("Player is Black")
            else:
                print("Error - pt.1")

            if np.random.rand() <= epsilon:
                action = np.random.randint(0, checkers_actions, size=1)[0]
                #print("Random")
            else:
                q_value = model.predict(state)                        # Q-values predictions
                action = np.argmax(q_value)                           # Move with highest Q-value is the chosen one
                #print("Predicted")

            #action_mapping = action_decision(action)
            #print(action_mapping)
            #moves = action_mapping.split(',')
            #print(str(int(moves[0])) + "-" + str(int(moves[1])) + "-")

            # Place in Action for the next board state
            #observation_new = checkers_game.move(checkers_game, player_turn, int(moves[0]), int(moves[1]))
            (source, dest) = action_decision(action)
            checkers_game.move(source, dest)
            observation_new = np.array(checkers_game.board)
            obs_new = np.expand_dims(np.asarray(observation_new), axis=0)  # (Formatting issues)
            state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :],
                                  axis=1)  # Update the input with the new state of the game

            # storing of moves for training
            if player_turn == checkers.Player.WHITE:
                if player_turn != last_player:
                    if new_game == True:
                        p1_stored_actions.append((state, action, 0, state, False))
                    else:
                        temp_list = list(p1_stored_actions[-1])
                        temp_list[3] = last_board_state  # update the previous move future state network
                        temp_list[2] = 1  # update the previous states score
                        p1_stored_actions[-1] = tuple(temp_list)
                        p1_stored_actions.append((state, action, 0, state, False))
                        # This will initialize the next states stored moves that will later come back to edit
                else:  # Means the input was invalid -- penalty needed
                    temp_list = list(p1_stored_actions[-1])
                    # temp_list[3] = game.board   # update the previous move future state network
                    temp_list[2] = -1  # update the previous states score
                    p1_stored_actions[-1] = tuple(temp_list)
                    p1_stored_actions.append((state, action, 0, state, False))
            elif player_turn == checkers.Player.BLACK:
                if player_turn != last_player:
                    if first_Black_Turn == True:
                        p2_stored_actions.append((state, action, 0, state, False))
                        first_Black_Turn = False
                    else:
                        temp_list = list(p2_stored_actions[-1])
                        temp_list[3] = last_board_state  # update the previous move future state network
                        temp_list[2] = 1  # update the previous states score
                        p2_stored_actions[-1] = tuple(temp_list)
                        p2_stored_actions.append((state, action, 0, state, False))
                        # This will initialize the next states stored moves that will later come back to edit
                else:  # Means the input was invalid -- penalty needed
                    temp_list = list(p2_stored_actions[-1])
                    # temp_list[3] = game.board   # update the previous move future state network
                    temp_list[2] = -1  # update the previous states score
                    p2_stored_actions[-1] = tuple(temp_list)
                    p2_stored_actions.append((state, action, 0, state, False))
            else:
                print("Error - pt.2")

            # Save state for next player
            state = state_new

            if player_turn == checkers.Player.WHITE:
                p1_state = state
                # print("Player is White")
            elif player_turn == checkers.Player.BLACK:
                p2_state = state
                # print("Player is Black")

            last_player = player_turn
            last_board_state = state
            new_game = False

            #print("The Len of White stored actions {}".format(len(p1_stored_actions)))
            #print("The Len of Black stored actions {}".format(len(p2_stored_actions)))
            #print('')

        # There should be enough collected to Train a network after leaving the previous loop
        # train_qvalue_nn(stored_actions, batch_size, model, gamma, checkers_actions, state)
        if len(p1_stored_actions) >= observe_time:
            p1_model = train_qvalue_nn(p1_stored_actions, batch_size, p1_model, gamma, checkers_actions, state)
            print("\nWhite Model Trained \n")
            p1_stored_actions = deque()
            p1_stored_actions.append((state, action, 0, state, False))
        elif len(p2_stored_actions) >= observe_time:
            p2_model = train_qvalue_nn(p1_stored_actions, batch_size, p2_model, gamma, checkers_actions, state)
            print("\nBlack Model Trained \n")
            p2_stored_actions = deque()
            p2_stored_actions.append((state, action, 0, state, False))
        else:
            print("WUUUUUUUT?!?!?!?!?!")
            print("No Model Training")

        checkers.board_print(checkers_game.board)



        print("\nYou have 5 seconds to input the following - (Ctrl-C to start / Ctrl-F2 in PyCharm IDE)")
        print("-) Type 'Save' to Save and quit")
        print("-) Type 'Escape' to quit")
        print("-) 5 second timeout to continue")
        # print("User input: ")
        try:
            for i in range(0,5):
                sleep(1)
        except KeyboardInterrupt:
            usr_input = input("User elected to input the following: ")

        if usr_input == "Save":
            # Save the model
            p1_model.save_weights('White_model_weights.h5')
            p2_model.save_weights('Black_model_weights.h5')
            usr_input = "Escape"
            print("Saving and Leaving the Game")
        elif usr_input == " Save":
            # Save the model
            p1_model.save_weights('White_model_weights.h5')
            p2_model.save_weights('Black_model_weights.h5')
            usr_input = "Escape"
            print("Saving and Leaving the Game")
        elif usr_input == "Escape":
            print("Leaving the Game")
        elif usr_input == " Escape:":
            # if input error
            usr_input = "Escape"
            print("Leaving the Game")
        else:
            #print("input was: -", usr_input)
            usr_input = ""
            print("Continuing game")

    return "Game Ended"


