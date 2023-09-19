import numpy as np
from numba import njit, jit
@njit
def w_fast_update(wfe, wfe_decay, wfi, wfi_decay, Neurot, w_fast, N, NE):
  alpha_w = np.empty(N, dtype=np.float32)
  alpha_w[:NE] = wfe * Neurot[:NE] - wfe_decay * w_fast[:NE]
  alpha_w[NE:] = wfi * Neurot[NE:] - wfi_decay * w_fast[NE:]
  return alpha_w