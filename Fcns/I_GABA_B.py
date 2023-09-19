import numpy as np
from numba import njit, jit
##Function of GABA B current (mA/cm^2).
@njit
def I_GABA_B(g_GABA_B, Vm):
  E_GABA_B = -95.0  #Reversal potential for GABA B channels (mV).
  return g_GABA_B * (Vm - E_GABA_B) / (1.0 + np.exp(-(Vm + 80.0) / 25.0))