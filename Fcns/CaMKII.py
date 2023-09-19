import numpy as np
from numba import njit, jit
##Function of fraction CaMKII bound to Ca2+.
#F: fraction of CaMKII subunits bound to Ca+ /CaM.
@njit
def CaMKII(Cai, Cai_eq): 
  K_H1 = 2  # The Ca2 activation Hill constant of CaMKII in uM.
  b = K_H1 - Cai_eq #Value to set the function as 0.5 at Cai_eq.
  return ((Cai + b) / K_H1)**4 / (1 + (((Cai + b) / K_H1)**4))