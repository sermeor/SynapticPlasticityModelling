import numpy as np
from numba import njit, jit

##Function of AMPA current (mA/cm^2).
@njit
def I_AMPA(g_AMPA, Vm):
  E_AMPA = 0.0  #Reversal potential for AMPA channels (mV)
  return g_AMPA * (Vm - E_AMPA)