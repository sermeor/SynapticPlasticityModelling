import numpy as np
from numba import njit, jit
#Function that updates the connectivity matrix.
@njit
def connectivity_update(C, act, N, NE, w_plas):
  # Postsynaptic update.
  sigma = 0.5 #Equilibrium activity of the neuron.
  w_ex = - 0.00002 #Weight of rate of change of post-synaptic connectivity of the neuron on excitatory plates (- when act>sigma, + when act<sigma, compensation).
  w_inh = 0.00001 #Weight of rate of change of post-synaptic connectivity of the neuron on inhibitory plates (+ when act>sigma, - when act<sigma, compensation).
  post_delta_C_row = (act - sigma)
  ##TESTING
  w_plas = 0
  ##TESTING
  post_delta_C = np.empty((N, N), dtype=np.float32)
  post_delta_C[:NE] = post_delta_C_row * (w_ex + w_plas)
  post_delta_C[NE:] = post_delta_C_row * (w_inh + w_plas)

  # Presynaptic update.
  w = 0.00001 #Weight of rate of change of presynaptic connectivity of the neuron (+ when act>sigma, - when act<sigma).
  pre_delta_C = np.empty((N, N), dtype=np.float32)
  for i in range(N):  #rows
    pre_delta_C[i, :] = w * (act[i] - sigma)

  # Sum both effects.
  delta_C = post_delta_C + pre_delta_C

  # Check it is in [0, 1]
  temp = C + delta_C
  invalid_indices = np.logical_or(temp < 0, temp > 1)
  delta_C = np.where(invalid_indices, 0, delta_C)

  return delta_C