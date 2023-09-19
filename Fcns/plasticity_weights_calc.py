import numpy as np
from numba import njit, jit
# Weights of BDNF/proBDNF and CaMKII plasticity,
#afecting the rate of change of the connectivity matrix.
#Sigmoid that can go to negative values.
@njit
def plasticity_weights_calc(CaMKII_bound, trkB_bound, p75NTR_bound):
  w_CaMKII_bound = 1 #Weight of CaMKII_bound
  w_trkB_bound = 1 #Weight of trkB_bound
  w_p75NTR_bound = 2 #Weight of p75NTR_bound
  s = 10 #Slope of sigmoid.
  w = 0.00001 #Weight to convert to plasticity from sigmoid.
  plasticity_weights = w/(1 + np.exp(-s*(w_CaMKII_bound*CaMKII_bound + w_trkB_bound*trkB_bound - w_p75NTR_bound*p75NTR_bound))) - 0.5
  return plasticity_weights

