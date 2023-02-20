## Function that simulates the injection of FMH into the peritoneum.
#t_start is the time for the first dose, and t_repeat is the repetition
#time in hours. inj_time is time that the injection lasts. Output units in ug/h. 


import numpy as np

def FMH_inj(t, t_start, t_repeat, q):
    inj_time = 1/3600
    n = len(t)
    f = np.zeros(n)
    for i in range(n):
        if t[i] > t_start:
            n_stim = int((t[i]-t_start)//t_repeat)
            if (t[i] > (t_start + n_stim * t_repeat)) and (t[i] <= (t_start + inj_time + n_stim * t_repeat)):
                f[i] = q/inj_time
            else:
                f[i] = 0
        else:
            f[i] = 0
    return f