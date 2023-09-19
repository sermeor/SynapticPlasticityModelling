import numpy as np
from numba import njit, jit
##Function of GABA A current (mA/cm^2).
@njit
def I_GABA_A(g_GABA_A, Vm):
  E_GABA_A = -70.0  #Reversal potential for GABA A channels (mV).
  return g_GABA_A * (Vm - E_GABA_A)