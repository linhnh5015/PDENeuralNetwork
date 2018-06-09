import json
import numpy as np

with open('F:/Hoang Linh/Nam3/Ky_2/stochastic_process/code/SequenceDataSet.json') as f:
    data = json.load(f)

count_n = 0
count_c = 0
n_to_n = 0
n_to_c = 0
c_to_n = 0
c_to_c = 0
n_emiting_A = 0
n_emiting_C = 0
n_emiting_G = 0
n_emiting_T = 0
c_emiting_A = 0
c_emiting_C = 0
c_emiting_G = 0
c_emiting_T = 0
for sequence in data:
    i = 0
    while i < len(sequence['Label']) - 1:
        this_label = sequence['Label'][i]
        next_label = sequence['Label'][i + 1]
        this_obs = sequence['Sequence'][i]
        if this_label == 'n':
            count_n += 1
            if next_label == 'n':
                n_to_n += 1
            else:
                n_to_c += 1
            if this_obs == 'A':
                n_emiting_A += 1
            elif this_obs == 'C':
                n_emiting_C += 1
            elif this_obs == 'G':
                n_emiting_G += 1
            else:
                n_emiting_T += 1
        else:
            count_c += 1
            if next_label == 'n':
                c_to_n += 1
            else:
                c_to_c += 1
            if this_obs == 'A':
                c_emiting_A += 1
            elif this_obs == 'C':
                c_emiting_C += 1
            elif this_obs == 'G':
                c_emiting_G += 1
            else:
                c_emiting_T += 1
        i += 1

rate_n_to_n = n_to_n/(n_to_n + n_to_c)
rate_n_to_c = n_to_c/(n_to_n + n_to_c)
rate_c_to_n = c_to_n/(c_to_n + c_to_c)
rate_c_to_c = c_to_c/(c_to_n + c_to_c)
rate_n_emiting_A = n_emiting_A/(n_emiting_A + n_emiting_C +n_emiting_G + n_emiting_T)
rate_n_emiting_C = n_emiting_C/(n_emiting_A + n_emiting_C +n_emiting_G + n_emiting_T)
rate_n_emiting_G = n_emiting_G/(n_emiting_A + n_emiting_C +n_emiting_G + n_emiting_T)
rate_n_emiting_T = n_emiting_T/(n_emiting_A + n_emiting_C +n_emiting_G + n_emiting_T)

rate_c_emiting_A = c_emiting_A/(c_emiting_A + c_emiting_C +c_emiting_G + c_emiting_T)
rate_c_emiting_C = c_emiting_C/(c_emiting_A + c_emiting_C +c_emiting_G + c_emiting_T)
rate_c_emiting_G = c_emiting_G/(c_emiting_A + c_emiting_C +c_emiting_G + c_emiting_T)
rate_c_emiting_T = c_emiting_T/(c_emiting_A + c_emiting_C +c_emiting_G + c_emiting_T)

rate_n = count_n/(count_n + count_c)
rate_c = count_c/(count_n + count_c)
transition = np.array([
    [rate_n_to_n,  rate_n_to_c],
    [rate_c_to_n, rate_c_to_c]])
emission = np.array([
    [rate_n_emiting_A,  rate_n_emiting_C, rate_n_emiting_G, rate_n_emiting_T],
    [rate_c_emiting_A,  rate_c_emiting_C, rate_c_emiting_G, rate_c_emiting_T]])
initial = np.array([rate_n, rate_c])
np.savetxt('F:/Hoang Linh/Nam3/Ky_2/stochastic_process/code/transition', transition)
np.savetxt('F:/Hoang Linh/Nam3/Ky_2/stochastic_process/code/emission', emission)
np.savetxt('F:/Hoang Linh/Nam3/Ky_2/stochastic_process/code/initial', initial)
