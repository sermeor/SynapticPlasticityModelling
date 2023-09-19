import numpy as np
from numba import njit, jit
##Function that draws a Bernoulli sample from the probability of neuron firing.
#Output is 1 (active) or 0 (inactive). Seed is fixed to have the same results.
@njit
def spike_boolean(Vm):
  Vth = 0
  result = np.empty_like(Vm, dtype=np.int32)
  for i in range(len(Vm)):
    result[i] = 1 if Vm[i] >= Vth else 0
  return result