## Function that simulates the injection of FMH into the peritoneum.
#t_start is the time for the first dose, and t_repeat is the repetition
#time in hours. inj_time is time that the injection lasts. Output units in ug/h. 

def FMH_inj(t, t_start, t_repeat, q):
  inj_time = 1/3600
  if t > t_start:
    n_stim = int((t - t_start)//t_repeat)
    if (t > (t_start + n_stim * t_repeat)) and (t <= (t_start + inj_time + n_stim * t_repeat)):
      f = q/inj_time
    else:
      f = 0
  else:
    f = 0
  return f