import numpy as np
from numba import njit, jit
##Function of leakage current (mA/cm^2).
@njit
def I_L(Vm):
  g_L = 0.3  # maximum leak conductance (mS/cm^2)
  E_L = -54.4  # leak reversal potential (mV)
  return g_L * (Vm - E_L)