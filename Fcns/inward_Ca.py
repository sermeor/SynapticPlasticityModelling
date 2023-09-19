import numpy as np
from numba import njit, jit
##Function of inward calcium rate (uM/ms).
@njit
def inward_Ca(g_NMDA, Vm, c):
  F = 96485  # Faraday Constant (mA*ms/umol).
  d = 8.4e-6  #Distance of membrane shell where calcium ions enter (cm).
  s = 1000  #conversor umol/(cm^3 * ms) to uM/ms.
  return -s * (I_NMDA_Ca(g_NMDA, Vm) + I_VGCC(c, Vm)) / (2 * F * d)