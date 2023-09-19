import numpy as np
from numba import njit, jit

## Function of sodium channel current (mA/cm^2).
@njit
def I_Na(m, h, Vm):
  g_Na = 120.0  # maximum sodium conductance (mS/cm^2)
  E_Na = 50.0  # sodium reversal potential (mV)
  return g_Na * (m**3) * h * (Vm - E_Na)