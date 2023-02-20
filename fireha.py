## Histamine firing function. 
# Units in events/h.
# 
import numpy as np

def fireha(t, i_factor):
    # Units in events/h.
    n = len(t)
    r = 150
    repeat_time = 60*10
    t_start = 5
    t_flip = 7
    t_end = 15
    basal = 1
    b = 2
    max_f = 10
    stim_boolean = 0

    f = np.zeros(n)

    if stim_boolean == 0:
        f = basal * i_factor
        f[f > max_f] = max_f
        return f

    for i in range(n):
        time = t[i] * 3600
        n_stim = int(np.floor(time / repeat_time))
        if time <= (t_start + n_stim * repeat_time):
            f[i] = basal * i_factor[i]
        elif time > (t_start + n_stim * repeat_time) and time <= (t_flip + n_stim * repeat_time):
            f[i] = i_factor[i] * (basal + r * (1 - np.exp(-b * ((time - n_stim * repeat_time) - t_start))))
        elif time < (t_end + n_stim * repeat_time):
            f[i] = i_factor[i] * (basal + r * (np.exp(-b * ((time - n_stim * repeat_time) - t_flip)) - np.exp(-b * (time - n_stim * repeat_time))))
        else:
            f[i] = i_factor[i] * basal

    f[f > max_f] = max_f
    return f