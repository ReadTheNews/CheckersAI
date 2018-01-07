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

    obs = np.expand_dims(np.asarray(checkers_game.board), axis=0)
    p1_state = np.stack((obs, obs), axis=1)
    p2_state = np.stack((obs, obs), axis=1)

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
    maximum_moves = 20000
    move_number_i = 0

    # initialize a few parameters
    old_board_W = checkers_game.board
    old_board_B = checkers_game.board
    reward = 1

    # Inside this function -- the game should run "Endlessly"

    while usr_input != "Escape":
        while len(p1_stored_actions) < observe_time and len(p2_stored_actions) < observe_time:

            player_turn = checkers_game.turn

            # checks if the game is over
            if checkers_game.is_over():
                new_game = True
                first_Black_Turn = True
                move_number_i = 0
                checkers_game.reset()
                print("\n###--- New Game! ---###")
                if checkers_game.turn == checkers.Player.WHITE:
                    print("### --- Black Won! ---###\n\n")

                    # Update Player 1 score / stored values
                    temp_list = list(p1_stored_actions[-1])
                    temp_list[3] = last_board_state  # update the previous move future state network
                    temp_list[2] = -1000  # update the previous states score
                    temp_list[4] = True  # update the previous states score
                    p1_stored_actions[-1] = tuple(temp_list)
                    # Update Player 2 score / stored values
                    temp_list = list(p2_stored_actions[-1])
                    temp_list[3] = last_board_state  # update the previous move future state network
                    temp_list[2] = 1000  # update the previous states score
                    temp_list[4] = True  # update the previous states score
                    p2_stored_actions[-1] = tuple(temp_list)


                elif checkers_game.turn == checkers.Player.BLACK:
                    print("### --- White Won! ---###\n\n")

                    # Update Player 1 score / stored values
                    temp_list = list(p1_stored_actions[-1])
                    temp_list[3] = last_board_state  # update the previous move future state network
                    temp_list[2] = 1000  # update the previous states score
                    temp_list[4] = True  # update the previous states score
                    p1_stored_actions[-1] = tuple(temp_list)
                    # Update Player 2 score / stored values
                    temp_list = list(p2_stored_actions[-1])
                    temp_list[3] = last_board_state  # update the previous move future state network
                    temp_list[2] = -1000  # update the previous states score
                    temp_list[4] = True  # update the previous states score
                    p2_stored_actions[-1] = tuple(temp_list)


            # checks if board is stale or maximum number of moves is met
            if checkers_game.is_stale() or move_number_i == maximum_moves:
                new_game = True
                first_Black_Turn = True
                move_number_i = 0
                checkers.board_print(checkers_game.board)
                checkers_game.reset()
                if checkers_game.is_stale():
                    print("\n###--- Stale Game! ---###")
                    if len(p1_stored_actions) > 0:
                        temp_list = list(p1_stored_actions[-1])
                        temp_list[3] = last_board_state  # update the previous move future state network
                        temp_list[2] = -100  # update the previous states score
                        temp_list[4] = True  # update the previous states score
                        p1_stored_actions[-1] = tuple(temp_list)
                    # Update Player 2 score / stored values
                    if len(p1_stored_actions) > 0:
                        temp_list = list(p2_stored_actions[-1])
                        temp_list[3] = last_board_state  # update the previous move future state network
                        temp_list[2] = -100  # update the previous states score
                        temp_list[4] = True  # update the previous states score
                        p2_stored_actions[-1] = tuple(temp_list)
                else:
                    print("$------ Max moves reached ------$")
                    if len(p1_stored_actions) > 0:
                        temp_list = list(p1_stored_actions[-1])
                        temp_list[3] = last_board_state  # update the previous move future state network
                        temp_list[2] = -1  # update the previous states score
                        temp_list[4] = False  # update the previous states score
                        p1_stored_actions[-1] = tuple(temp_list)
                    # Update Player 2 score / stored values
                    if len(p2_stored_actions) > 0:
                        temp_list = list(p2_stored_actions[-1])
                        temp_list[3] = last_board_state  # update the previous move future state network
                        temp_list[2] = -1  # update the previous states score
                        temp_list[4] = False  # update the previous states score
                        p2_stored_actions[-1] = tuple(temp_list)
                # Update Player 1 score / stored values



            if player_turn == checkers.Player.WHITE:
                model = p1_model
                state = p1_state
                #print("Player is White")
            elif player_turn == checkers.Player.BLACK:
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

            (source, dest) = action_decision(action)

            if player_turn == checkers.Player.WHITE:
                old_board_W = checkers_game.board
            elif player_turn == checkers.Player.BLACK:
                old_board_B = checkers_game.board

            checkers_game.move(source, dest)
            observation_new = np.array(checkers_game.board)
            obs_new = np.expand_dims(np.asarray(observation_new), axis=0)  # (Formatting issues)
            state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :],
                                  axis=1)  # Update the input with the new state of the game

            # Analyzing board jumps and move rewards


            # storing of moves for training
            if player_turn == checkers.Player.WHITE:
                if player_turn != last_player:
                    if new_game == True:
                        p1_stored_actions.append((state, action, 0, state, False))
                    else:
                        temp_list = list(p1_stored_actions[-1])
                        # opposing player score
                        if len(p2_stored_actions) > 0:
                            compare_temp_list = list(p2_stored_actions[-1])
                            if compare_temp_list[2] == 150:
                                temp_list[2] = 20  # update the previous states score, player got jumped

                        temp_list[3] = last_board_state  # update the previous move future state network
                        temp_list[2] = 50  # update the previous states score
                        p1_stored_actions[-1] = tuple(temp_list)

                        # checks if king or jump then will reward accordingly
                        if checkers.move_jumps(old_board_W, player_turn, source, dest):
                            reward = 150
                        elif checkers.move_grants_king(player_turn, dest):
                            reward = 100
                        else:
                            reward = 50
                        p1_stored_actions.append((state, action, reward, state, False))

                else:  # Means the input was invalid -- penalty needed
                    temp_list = list(p1_stored_actions[-1])
                    # temp_list[3] = game.board   # update the previous move future state network
                    temp_list[2] = -10  # update the previous states score
                    p1_stored_actions[-1] = tuple(temp_list)


                    p1_stored_actions.append((state, action, 0, state, False))
            elif player_turn == checkers.Player.BLACK:
                if player_turn != last_player:
                    if first_Black_Turn == True:
                        p2_stored_actions.append((state, action, 0, state, False))
                        first_Black_Turn = False
                    else:
                        temp_list = list(p2_stored_actions[-1])
                        #opposing player score
                        if len(p1_stored_actions) > 0:
                            compare_temp_list = list(p1_stored_actions[-1])
                            if compare_temp_list[2] == 150:
                                temp_list[2] = 20  # update the previous states score, player got jumped

                        temp_list[3] = last_board_state  # update the previous move future state network
                        p2_stored_actions[-1] = tuple(temp_list)

                        # checks if king or jump then will reward accordingly
                        if checkers.move_jumps(old_board_B, player_turn, source, dest):
                            reward = 150
                        elif checkers.move_grants_king(player_turn, dest):
                            reward = 100
                        else:
                            reward = 50
                        p2_stored_actions.append((state, action, reward, state, False))
                        # This will initialize the next states stored moves that will later come back to edit
                else:  # Means the input was invalid -- penalty needed
                    temp_list = list(p2_stored_actions[-1])
                    # temp_list[3] = game.board   # update the previous move future state network
                    temp_list[2] = -10  # update the previous states score
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
            move_number_i = move_number_i + 1

        # There should be enough collected to Train a network after leaving the previous loop
        if len(p1_stored_actions) >= observe_time:
            p1_model = train_qvalue_nn(p1_stored_actions, batch_size, p1_model, gamma, checkers_actions, state)
            print("\nWhite Model Trained \n")
            p1_stored_actions = deque()
            p1_stored_actions.append((state, action, 0, state, False))
        elif len(p2_stored_actions) >= observe_time:
            p2_model = train_qvalue_nn(p2_stored_actions, batch_size, p2_model, gamma, checkers_actions, state)
            print("\nBlack Model Trained \n")
            p2_stored_actions = deque()
            p2_stored_actions.append((state, action, 0, state, False))
        else:
            print("WUUUUUUUT?!?!?!?!?!")
            print("No Model Training")

        checkers.board_print(checkers_game.board)



        print("\nYou have 3 seconds to input the following - (Ctrl-C to start / Ctrl-F2 in PyCharm IDE)")
        print("-) Type 'Save' to Save and quit")
        print("-) Type 'Escape' to quit")
        print("-) 3 second timeout to continue")
        # print("User input: ")
        try:
            for i in range(0,3):
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
            print("\nContinuing game")

    return "Game Ended"


