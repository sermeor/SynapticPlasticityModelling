import numpy as np

from inhibsynHAtoHA import *
from VMATH import *
from VHNMT import *
from VHNMTg import *
from VHTDC import *
from VHTDCg import *
from VHAT import *
from VHATg import *
from VHTL import *
from VHTLg import *
from H1ha import *
from HTin import *
from inhibRHAtoHA import *
from fireha import *





def comp_model(z, t):
  
  dz = np.zeros(len(z)) #Initialization of differential terms.
  
  ## ------------------Histamine neuron model -------------------------------
  ## Histamine neuron variables
  #z[0] Cytosolic histamine (uM)
  #z[1] Vesicular histamine (uM)
  #z[2] Extracellular histamine (uM)
  #z[3] Blood histidine (uM)
  #z[4] Cytosolic histidine (uM)
  #z[5] Cytosolic histidine pool (uM)
  #z[6] Activated g-coupled protein H3 receptor (uM)
  #z[7] Activated T protein H3 receptor (uM)
  #z[8] Histamine bound to h3 receptor (uM)
  #z[9] Histamine in glia (uM)
  #z[10] Histidine in glia (uM)
  #z[11] Pool of histidine in glia (uM)
  

  ## Histamine neuron constants. 
  b1 = 15  #HA leakage from the cytosol to the extracellular space.
  b2 = 0.75*3.5  #HA release per action potential.
  b3 = 0.05  #HA removal from the extracellular space
  b4 = .25  #Strength of stabilization of bHT to bHT0.
  b5 = 2.5 #From cHT to HTpool.
  b6 = 1 #From HTpool to cHT.
  b7 = 1 #Other uses of HT remove HT.
  b8 = 100 #Histamine bound to autoreceptors produce G∗.
  b9 = 961.094 #T∗ facilitates the reversion of G∗ to G. X
  b10 = 20 #G∗ produces T∗. X
  b11 = 66.2992 #decay coefficient of T∗ X
  b12 = 5  #eHA binds to autoreceptors. X
  b13 = 65.61789 #eHA dissociates from autoreceptors X
  b14 = 15 #eHA - gHA leakage. 
  b15 = 1 #From gHT to gHTpool.
  b16 = 1 # From gHTpool to gHT.
  b17 = 1 #Removal of gHT or use somewhere else.
  g0HH = 10  #Total g-coupled protein for H3 on HA neuron
  t0HH = 10 #Total T protein for H3 on HA neuron
  b0HH = 10  #Total bound H3 receptors on HA neuron
  
  
  #Steady state values.
  gstar_ha_basal =  0.7484 #Equilibrium concentration of g* histamine in H3 receptor.
  bht0 = 100 #Steady state value of blood histidine.
  

  dz[0] = inhibsynHAtoHA(z[6], gstar_ha_basal) * VHTDC(z[4])  - VMATH(z[0], z[1]) -  VHNMT(z[0]) - b1*(z[0] - z[2]) + VHAT(z[2])
  dz[1] = VMATH(z[0], z[1]) - inhibRHAtoHA(z[6], gstar_ha_basal)*fireha(t)*b2*z[1]
  dz[2] = inhibRHAtoHA(z[6], gstar_ha_basal)*fireha(t)*b2*z[1] - VHAT(z[2]) + b1*(z[0] - z[2])  - b3*z[2] - H1ha(z[2])*VHATg(z[2]) + b14*(z[9] - z[2])
  dz[3] = HTin(t) - VHTL(z[3])  - b4*(z[3] - bht0) 
  dz[4] = VHTL(z[3]) - inhibsynHAtoHA(z[6], gstar_ha_basal) *  VHTDC(z[4]) - b5*z[4] + b6*z[5]
  dz[5] = b5*z[4] - b6*z[5] - b7*z[5]
  dz[6]  = b8*z[8]**2*(g0HH - z[6]) - b9*z[7]*z[6]
  dz[7] = b10*z[6]**2*(t0HH - z[7])  - b11*z[7]
  dz[8] = b12*z[2]*(b0HH - z[8])  - b13*z[8]
  dz[9] = H1ha(z[2])*VHATg(z[2]) - b14*(z[9] - z[2]) - VHNMTg(z[9]) + VHTDCg(z[10])
  dz[10] = - VHTDCg(z[10]) + VHTLg(z[3]) - b15*z[10] + b16*z[11]
  dz[11] =  b15*z[10] - b16*z[11] - b17*z[11]
  
  

  return dz