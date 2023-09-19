import numpy as np
from numba import njit, jit
##Function of potassium channels current (mA/cm^2).
@njit
def I_K(n, Vm):
  E_K = -77.0  # potassium reversal potential (mV)
  g_K = 36.0  # maximum potassium conductance (mS/cm^2)
  return g_K * (n**4) * (Vm - E_K)