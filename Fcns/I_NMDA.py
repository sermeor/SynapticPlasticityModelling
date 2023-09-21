import numpy as np
from numba import njit, jit
from Fcns.I_NMDA_Na import *
from Fcns.I_NMDA_K import *
from Fcns.I_NMDA_Ca import *
##Function of total NMDA channel current density (mA/cm^2).
@njit
def I_NMDA(g_NMDA, Vm):
  return I_NMDA_Na(g_NMDA, Vm) + I_NMDA_K(g_NMDA, Vm) + I_NMDA_Ca(g_NMDA, Vm)

