from keras.models import Sequential     # One Layer after the other
from keras.layers import Dense, Flatten          # Dense layers are fully connected layers

# ----------------------------------
# Create network. Input is two consecutive game states, output is Q-values of the possible moves.
# ----------------------------------

def qvalue_nn(checkers_actions):
    model = Sequential()
    model.add(Dense(64, input_dim=2, kernel_initializer='uniform', activation='relu'))
    model.add(Flatten())   # Flatten input so as to have no problems with processing
    model.add(Dense(200, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(400, kernel_initializer='uniform', activation='relu'))
    # Next layer - same number of outputs as possible actions
    model.add(Dense(checkers_actions, kernel_initializer='uniform', activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model
