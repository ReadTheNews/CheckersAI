from keras.models import Sequential     # One Layer after the other
from keras.layers import Dense, Flatten          # Dense layers are fully connected layers

# ----------------------------------
# Create network. Input is two consecutive game states, output is Q-values of the possible moves.
# ----------------------------------

def qvalue_nn(checkers_actions):
    model = Sequential()
    model.add(Dense(128, input_shape=(2,) + (64,), kernel_initializer='uniform', activation='relu'))
    # Shape = 2 sets of vectors of 64 (AKA - current game board, following action and repose board)
    model.add(Flatten())   # Flatten input so as to have no problems with processing
    model.add(Dense(500, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1000, kernel_initializer='uniform', activation='relu'))
    #model.add(Dense(2500, kernel_initializer='uniform', activation='relu'))
    # Next layer - same number of outputs as possible actions
    model.add(Dense(checkers_actions, kernel_initializer='uniform', activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model
