import numpy as np
from numba import njit, jit
##Function of magnesium block of NMDA dependent on voltage (mV).
@njit
def MgB(Vm):
  Mg0 = 2  #mM
  return 1 / (1 + (Mg0 * np.exp(-0.062 * Vm)) / 3.57)
