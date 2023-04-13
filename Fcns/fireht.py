# Function of serotonin neuron firing.
#Commented functions are different firing paradigms. 
# UNITS of f() in events/h, time variables in seconds. 

import numpy as np

def fireht(t, i_factor):
  # UNITS of f() in events/h, time variables in seconds.
  r = 8
  repeat_time = 60*10
  t_start = 5
  t_flip = 7
  t_end = 15
  basal = 1
  b = 2
  max_f = 1.5
  stim_boolean = 0 # 1 if stimulation activated
  
  
  if stim_boolean == 0:
    f = basal * i_factor
    if f>max_f:
      f = max_f
    return f
  
  
  time = t * 3600
  n_stim = int(np.floor(time / repeat_time))
  if time <= (t_start + n_stim * repeat_time):
    f = basal * i_factor
  elif time > (t_start + n_stim * repeat_time) and time <= (t_flip + n_stim * repeat_time):
    f = i_factor * (basal + r * (1 - np.exp(-b * ((time - n_stim * repeat_time) - t_start))))
  elif time < (t_end + n_stim * repeat_time):
    f = i_factor * (basal + r * (np.exp(-b * ((time - n_stim * repeat_time) - t_flip)) - np.exp(-b * (time - n_stim * repeat_time))))
  else:
    f = i_factor * basal
  if f>max_f:
    f = max_f
  return f