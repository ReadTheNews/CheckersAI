# CheckersAI
A few reads before going forward:
- [Reinforcement learning for complex goals using TF](https://www.oreilly.com/ideas/reinforcement-learning-for-complex-goals-using-tensorflow)
- [Demystifying Deep Reinforcement learning](https://www.intelnervana.com/demystifying-deep-reinforcement-learning/)
- [Reinforcement Learning ipython example - Big](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/16_Reinforcement_Learning.ipynb)
- [Reinforcement Learning ipython example - Small](https://github.com/llSourcell/deep_q_learning/blob/master/03_PlayingAgent.ipynb)
## Setup

If you don't already have [pipenv](https://docs.pipenv.org/basics/) installed, then ```pip install pipenv```

```
pipenv install
pipenv run python checkers.py
```
## Pre-Read

### Reinforcement learning for complex goals
- An agent receives a state (s) from the environment
- Produces an action (a)
#### Given this state and action pair, the environment provides:
- A new state (s′)
- A reward (r)

### Reinforcement learning using Q-learning
- We learn a direct mapping between state and action pairs (s,a) and value estimations (v)
- The value estimation should correspond to the discounted expected reward over time from taking action (a) while in state (s)
- Using the Bellman equation, we can iteratively update our estimations of Q(s,a) for all possible state action pairs

### The Q-value formula
- Q(current state (s), action) = expected reward (r) + Future reward discount (gamma) * maximum of action s on Q(next state (s'), action (a) )

### How it learns?
- The Neural Network predicts the Q values for determining the next best action
- (Epsilon Greedy) Makes the choice of choosing the Highest Q-value action or allows for exploration


## How it works?
- Initially calls qvalue_nn function to create Qvalue formula
- User is prompted with 'y','n' to load in a previous model
-- This loads in .h5 NN weights to reduce storage size of the model
-- No training data is ever stored
- Currently only allows user to watch training of the network
--- In the works: put in an ability to watch the networks predict (no training, just witnessing moves)

#### Qvalue_NN
- Function call to create fully connected 3-layer NN
-- Layers of 128,250,400 that outputs 4096 options (8^4 possible moves)
-- This function is the Q-value equation that will be used to predict moves

#### Action_collect
- Function call that goes through the process of allowing the 2 player models to train
-- action_collect will then store: board states, actions, rewards for each corresponding player (i.e. Player's model)
-- Then after enough samples are created for the model, through prediction or epsilon chance of a random move, will update NN weights for the corresponding model
-- Allows user to 'Save' (Save models and quit) or 'Quit' (No saving and exits program
- The program will run until user keyboard interrupts (Ctrl-C or Ctrl-F2 if using IDE like pycharm) when prompted

#### Action_decision
- Currently it is only a function using modulus math to create a move out of the number
-- In the works: passing back a reward through this function at some point (rewards for jumping an apposing player and penalty for being jumped)
