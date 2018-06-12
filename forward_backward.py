from builtins import enumerate

from hmm import HMM
import numpy as np
import json


def forward_backward(obs, transition, emission, initial):  # returns model given the initial model and observations
    num_states = len(initial)
    fwd = [[0 for i in range(num_states)] for j in range(len(obs))]
    # Initialize base cases (t == 0)
    for y in range(num_states):
        fwd[0][y] = initial[y] * emission[y][obs[0]]
    # Run Forward algorithm for t > 0
    for t in range(1, len(obs)):
        for y in range(num_states):
            fwd[t][y] = sum((fwd[t - 1][y0] * transition[y0][y] * emission[y][obs[t]]) for y0 in range(num_states))
    bwk = [[0 for i in range(num_states)] for j in range(len(obs))]
    T = len(obs)
    # Initialize base cases (t == T)
    for y in range(num_states):
        bwk[T - 1][y] = 1
    for t in reversed(range(T - 1)):
        for y in range(num_states):
            bwk[t][y] = sum((bwk[t + 1][y1] * transition[y][y1] * emission[y1][obs[t + 1]]) for y1 in range(num_states))
    p_obs = sum((initial[y] * emission[y][obs[0]] * bwk[0][y]) for y in range(num_states))
    gamma = [[0 for i in range(num_states)] for j in range(len(obs))]
    zi = [[[0 for i in range(num_states)] for j in range(num_states)] for k in range(len(obs))]

    for t in range(len(obs)):
        for y in range(num_states):
            gamma[t][y] = (fwd[t][y] * bwk[t][y]) / p_obs
            if t == 0:
                initial[y] = gamma[t][y]
            # compute zi values up to T - 1
            if t == len(obs) - 1:
                continue
            for y1 in range(num_states):
                zi[t][y][y1] = fwd[t][y] * transition[y][y1] * emission[y1][obs[t + 1]] * bwk[t + 1][y1] / p_obs

    for y in range(num_states):
        for y1 in range(num_states):
                # we will now compute new a_ij
            val = sum([zi[t][y][y1] for t in range(len(obs) - 1)])  #
            val /= sum([gamma[t][y] for t in range(len(obs) - 1)])
            transition[y][y1] = val
        # re estimate gamma
    for y in range(num_states):
        num_obs = emission.shape[1]
        for k in range(num_obs):  # for all symbols vk
            val = 0.0
            for t in range(len(obs)):
                if obs[t] == k:
                    val += gamma[t][y]
            val /= sum([gamma[t][y] for t in range(len(obs))])
            emission[y][k] = val
    return transition, emission, initial


if __name__ == "__main__":
    transmat = np.loadtxt('transition')
    emmat = np.loadtxt('emission')
    initial = [np.loadtxt('initial')][0]
    with open('TestDataSet.json') as f:
        data = json.load(f)
    hmm = HMM(transmat, emmat)
    observations = []
    for c in data[1]['Sequence']:
        if c == 'A':
            observations.append(0)
        elif c == 'C':
            observations.append(1)
        elif c == 'G':
            observations.append(2)
        else:
            observations.append(3)
    max_ite = 100
    count = 0
    epsilon = 0.1
    err = 9999
    while err > epsilon and count < max_ite:
        transition0, emission0, initial0 = forward_backward(observations, transmat, emmat, initial)
        err = np.linalg.norm(transmat - transition0, np.inf)
        err += np.linalg.norm(emmat - emission0, np.inf)
        err += np.linalg.norm(initial - initial0, np.inf)
        transmat = transition0
        emmat = emission0
        initial = initial0
