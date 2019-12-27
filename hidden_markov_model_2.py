# Source: http://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017
# Accessed: 2019-09-05

import numpy as np
import pandas as pd

# create state space and initial state probabilities
states = ['sleeping', 'eating', 'pooping']
hidden_states = ['healthy', 'sick']
pi = [0.5, 0.5]
state_space = pd.Series(pi, index=hidden_states, name='states')
print(state_space)
print('\n', state_space.sum())

# create hidden transition matrix
# a or alpha
#   = transition probability matrix of changing states given a state
# matrix is size (M x M) where M is number of states
a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
a_df.loc[hidden_states[0]] = [0.7, 0.3]
a_df.loc[hidden_states[1]] = [0.4, 0.6]

print(a_df)

a = a_df.values
print('\n', a, a.shape, '\n')
print(a_df.sum(axis=1))

# create matrix of observation (emission) probabilities
# b or beta = observation probabilities given state
# matrix is size (M x O) where M is number of states
# and O is number of different possible observations
observable_states = states

b_df = pd.DataFrame(columns=observable_states, index=hidden_states)
b_df.loc[hidden_states[0]] = [0.2, 0.6, 0.2]
b_df.loc[hidden_states[1]] = [0.4, 0.1, 0.5]

print(b_df)

b = b_df.values
print('\n', b, b.shape, '\n')
print(b_df.sum(axis=1))

# observation sequence of dog's behaviors
# observations are encoded numerically
obs_map = {'sleeping': 0, 'eating': 1, 'pooping': 2}
obs = np.array([1, 1, 2, 1, 0, 1, 2, 1, 0, 2, 2, 0, 1, 0, 1])

inv_obs_map = dict((v, k) for k, v in obs_map.items())
obs_seq = [inv_obs_map[v] for v in list(obs)]

print(pd.DataFrame(np.column_stack([obs, obs_seq]),
                   columns=['Obs_code', 'Obs_seq']))


# define Viterbi algorithm for shortest path
# code adapted from Stephen Marsland's, Machine Learning An Algorthmic Perspective, Vol. 2
# https://github.com/alexsosn/MarslandMLAlgo/blob/master/Ch16/HMM.py
def viterbi(pi, a, b, obs):
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]

    # init blank path
    path = np.zeros(T)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T))
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T))

    # init delta and phi
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    print('\nStart Walk Forward\n')
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t - 1] * a[:, s]) * b[s, obs[t]]
            phi[s, t] = np.argmax(delta[:, t - 1] * a[:, s])
            print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))

    # find optimal path
    print('-' * 50)
    print('Start Backtrace\n')
    path[T - 1] = np.argmax(delta[:, T - 1])
    print(phi)
    for t in range(T - 2, -1, -1):
        print(t + 1, path[t + 1])
        path[t] = phi[int(path[t + 1]), t + 1]
        print('path[{}] = {}'.format(t, path[t]))

    return path, delta, phi


path, delta, phi = viterbi(pi, a, b, obs)
print('\nsingle best state path: \n', path)
print('delta:\n', delta)
print('phi:\n', phi)

# result
state_map = {0: 'healthy', 1: 'sick'}
state_path = [state_map[v] for v in path]

print(pd.DataFrame()
      .assign(Observation=obs_seq)
      .assign(Best_Path=state_path))