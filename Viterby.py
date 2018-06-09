from hmm import HMM
import numpy as np
import json


def viterbi(hmm, initial_dist, emissions):
    probs = hmm.emission_dist(emissions[0]) * initial_dist
    stack = []

    for emission in emissions[1:]:
        trans_probs = hmm.transition_probs * np.row_stack(probs)
        max_col_ixs = np.argmax(trans_probs, axis=0)
        probs = hmm.emission_dist(emission) * trans_probs[max_col_ixs, np.arange(hmm.num_states)]

        stack.append(max_col_ixs)

    state_seq = [np.argmax(probs)]

    while stack:
        max_col_ixs = stack.pop()
        state_seq.append(max_col_ixs[state_seq[-1]])

    state_seq.reverse()
    s = ""
    for i in state_seq:
        if i == 0:
            s += 'n'
        else:
            s += 'c'

    return s


if __name__ == "__main__":
    replacements = []
    count = 0
    neg_count = 0
    pos_count = 0
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    transmat = np.loadtxt('F:/Hoang Linh/Nam3/Ky_2/stochastic_process/code/transition')
    emmat = np.loadtxt('F:/Hoang Linh/Nam3/Ky_2/stochastic_process/code/emission')
    initial = np.loadtxt('F:/Hoang Linh/Nam3/Ky_2/stochastic_process/code/initial')
    hmm = HMM(transmat, emmat)
    print(viterbi(hmm,initial, [3, 2, 1, 3, 0, 0, 2, 0, 1, 2, 3]))
    # with open('F:/Hoang Linh/Nam3/Ky_2/stochastic_process/code/TestDataSet.json') as f:
    #     data = json.load(f)
    # for sequence in data:
    #     observations = []
    #     for c in sequence['Sequence']:
    #         if c == 'A':
    #             observations.append(0)
    #         elif c == 'C':
    #             observations.append(1)
    #         elif c == 'G':
    #             observations.append(2)
    #         else:
    #             observations.append(3)
    #     obs = np.array(observations)
    #     result = viterbi(hmm, initial, obs)
    #     for i in range(0, len(sequence['Label']) - 1):
    #         count += 1
    #         if sequence['Label'][i] == 'n':
    #             neg_count += 1
    #             if result[i] == 'n':
    #                 TN += 1
    #             else:
    #                 FP += 1
    #         else:
    #             pos_count += 1
    #             if result[i] == 'n':
    #                 FN += 1
    #             else:
    #                 TP += 1
    #     print(sequence['Label'])
    #     print(result)
    #     print("##")
    # precision = TP/(TP + FP)
    # recall = TP/(TP + FN)
    # accuracy = (TP + TN)/(TP + TN + FP + FN)
    # f1 = 2*precision*recall/(precision + recall)
    # print(precision)
    # print(recall)
    # print(accuracy)
    # print(f1)
