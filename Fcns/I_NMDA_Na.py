import numpy as np
from numba import njit, jit
from Fcns.MgB import *
##Function of NMDA channel sodium current density (mA/cm^2).
@njit
def I_NMDA_Na(g_NMDA, Vm):
  P_Na = 1  #Permeability ratio to sodium.
  c = 0.1  #Conversor A/m^2 -> mA/cm^2
  P_NMDA = 10 * 10**(-9)  #m/s
  F = 96485  #C/mol
  R = 8.314  #J/K*mol
  T = 308.15  #K
  Nai = 18  #mM
  Nao = 140  #mM
  V_lim = 100  #mV
  a1 = g_NMDA * c * P_NMDA * P_Na * MgB(Vm) * ((Vm / 1000 * F**2) / (R * T))
  a2 = (Nai - Nao * np.exp(-((Vm / 1000 * F) / (R * T)))) / (1 - np.exp(-((Vm / 1000 * F) / (R * T))))

  a2[Vm > V_lim] = Nai
  a2[Vm < -V_lim] = Nao

  I = a1 * a2

  return I