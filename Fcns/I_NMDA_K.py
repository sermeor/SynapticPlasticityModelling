import numpy as np
from numba import njit, jit
from Fcns.MgB import *
##Function of NMDA channel potassium current density (mA/(cm^2 m-1))
@njit
def I_NMDA_K(g_NMDA, Vm):
  P_K = 1  #Permeability ratio to potassium.
  c = 0.1  #Conversor A/m^2 -> mA/cm^2
  P_NMDA = 10 * 10**(-9)  #m/s
  F = 96485  #C/mol
  R = 8.314  #J/K*mol
  T = 308.15  #K
  Ki = 140  #mM
  Ko = 5  #mM
  V_lim = 100  #mV
  a1 = g_NMDA * c * P_NMDA * P_K * MgB(Vm) * ((Vm / 1000 * F**2) / (R * T))
  a2 = (Ki - Ko * np.exp(-((Vm / 1000 * F) / (R * T)))) / (1 - np.exp(-((Vm / 1000 * F) / (R * T))))
  a2[Vm > V_lim] = Ki
  a2[Vm < -V_lim] = Ko

  I = a1 * a2

  return I