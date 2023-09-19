import numpy as np
from numba import njit, jit
##Function of fraction of p75_NTR proBDNF-bound (sigmoid).
@njit
def p75_NTR(bdnf, pro_bdnf):
  w_pro_bdnf = 0.9 #Factor of pro_bdnf.
  w_bdnf = 1 -  w_pro_bdnf #Factor of bdnf.
  s = 10 #Slope of sigmoid.
  return 1 / (1 + np.exp(-s*(w_bdnf * pro_bdnf + w_pro_bdnf * bdnf - 0.5)))