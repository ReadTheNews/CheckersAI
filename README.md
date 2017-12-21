# CheckersAI
A few reads before going forward:
- https://www.oreilly.com/ideas/reinforcement-learning-for-complex-goals-using-tensorflow
- https://www.intelnervana.com/demystifying-deep-reinforcement-learning/
- https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/16_Reinforcement_Learning.ipynb


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

## The Q-value formula
- Q(current state (s), action) = expected reward (r) + Future reward discount (gamma) * maximum of action s on Q(next state (s'), action (a) )

### How it learns?
- The Neural Network predicts the Q values for determining the next best action
- (Epsilon Greedy) Makes the choice of choosing the Highest Q-value action or allows for exploration
