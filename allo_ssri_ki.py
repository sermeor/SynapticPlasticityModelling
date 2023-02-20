## Function that models slow down in escit dissociation from SERTs due to allosteric binding
#Units in uM.
import numpy as np

def allo_ssri_ki(ssri):
  min_ki = 0.001
  b = 4 #Strength.
  ki = 0.05*np.exp(-b*ssri) + min_ki #constant of inhibition of ESCIT in uM. 
  return ki