import random

from keras.models import Sequential
from keras.layers import Dense, Flatten
import numpy as np
from collections import deque

import checkers

# (try_again, score, board)
def step(game, action):
    old_board = game.board
    old_turn = game.turn

    (source, dest) = divmod(action, checkers.BOARD_SIZE)
    if not game.move(source, dest):
        return True, -1, game.board
    elif checkers.move_jumps(old_board, old_turn, source, dest):
        return False, 1, game.board
    else:
        return False, 0, game.board


def turn(game, model, actions):
    try_again = True
    state = np.array(game.board)
    attempts = 0
    while try_again:
        attempts += 1
        if attempts > 10000:
            raise Exception('too many attempts')

        if np.random.rand() <= epsilon:
            action = np.random.randint(0, checkers.ACTION_SPACE, size=1)[0]
        else:
            q_value = model.predict(np.stack((state,)))           # Q-values predictions
            action = np.argmax(q_value)                           # Move with highest Q-value is the chosen one

        try_again, reward, state_new = step(game, action)
        actions.append((state, action, reward, np.array(state_new)))


def action_collect(p1_model, p2_model, epsilon, turns):
    p1_actions = deque()
    p2_actions = deque()

    game = checkers.Game()

    try:
        for i in range(turns):
            print('turn {}'.format(i))
            player_turn = game.turn
            if player_turn == checkers.Player.WHITE:
                turn(game, p1_model, p1_actions)
            else:
                turn(game, p2_model, p2_actions)

            if game.is_over():
                game.reset()
    except:
        checkers.board_print(game.board)

        raise

    return p1_actions, p2_actions


def model_create():
    model = Sequential()
    model.add(Dense(20, input_shape=(checkers.BOARD_SIZE,), init='uniform', activation='relu'))
    #model.add(Flatten())       # Flatten input so as to have no problems with processing
    model.add(Dense(18, init='uniform', activation='relu'))
    model.add(Dense(10, init='uniform', activation='relu'))
    model.add(Dense(checkers.ACTION_SPACE, init='uniform', activation='linear'))    # Same number of outputs as possible actions
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


p1_model = model_create()
p2_model = model_create()

turns = 500
epsilon = 0.7                              # Probability of doing a random move
gamma = 0.9                                # Discounted future reward. How much we care about steps further in time
mb_size = 50                               # Learning minibatch size

p1_actions, p2_actions = action_collect(p1_model, p2_model, epsilon, turns)
for action in p1_actions:
    print('{} {}'.format(action[1], action[2]))
