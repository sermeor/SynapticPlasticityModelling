import numpy as np
from numba import njit, jit
##Function that calculates the w_fast uptake.
@njit
def w_slow_update(wse, wse_decay, wsi, wsi_decay, Neurot, w_slow, N, NE):
  alpha_w = np.empty(N, dtype=np.float32)
  alpha_w[:NE] = wse * Neurot[:NE] - wse_decay * w_slow[:NE]
  alpha_w[NE:] = wsi * Neurot[NE:] - wsi_decay * w_slow[NE:]
  return alpha_w