import numpy as np
from numba import njit, jit
from Fcns.MgB import *
##Function of NMDA channel calcium current density (mA/cm^2).
@njit
def I_NMDA_Ca(g_NMDA, Vm):
  P_Ca = 10.6  #Permeability ratio to calcium.
  c = 0.1  #Conversor A/m^2 -> mA/cm^2
  P_NMDA = 10 * 10**(-9)  #m/s
  F = 96485  #C/mol
  R = 8.314  #J/K*mol
  T = 308.15  #K
  Cai = 0.0001  #mM
  Cao = 2  #mM
  V_lim = 100  #mV
  a1 = g_NMDA * c * P_NMDA * P_Ca * MgB(Vm) * ((4 * Vm / 1000 * F**2) /(R * T))
  a2 = (Cai - Cao * np.exp(-((2 * Vm / 1000 * F) / (R * T)))) / (1 - np.exp(-((2 * Vm / 1000 * F) / (R * T))))

  a2[Vm > V_lim] = Cai
  a2[Vm < -V_lim] = Cao

  I = a1 * a2

  return I