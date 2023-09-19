import numpy as np
from numba import njit, jit
##Function of BDNF presence dependent on neuronal factors (sigmoid).
@njit
def bdnf_calc(bAP_events, g_AMPA, Cai, Cai_eq, eht, eht_eq):
  delta_Cai = Cai - Cai_eq
  delta_eht = eht - eht_eq
  w_bAP = 0.1 #Weight of bAP on BDNF release.
  w_g_AMPA = 0.1 #Weight of g_AMPA on BDNF release.
  w_Cai = 0.1 #Weight of Cai increase above equilibrium on BDNF release.
  w_eht = 0.1 #Weight of eht increase above equilibrium on BDNF release.
  s = 10 #Slope of the sigmoid.
  bdnf = 1 / (1 + np.exp(-s*(w_bAP*bAP_events + w_g_AMPA*g_AMPA + w_Cai*delta_Cai + w_eht * delta_eht)))
  return bdnf