import numpy as np
from numba import njit, jit
##Function of VGCC current.
@njit
def I_VGCC(c, Vm):
  E_Ca = 60  #Reversal potential (mV)
  g_Ca = 0.0000075  # Maximum calcium conductance (mS/cm^2)
  return g_Ca * c * (Vm - E_Ca)