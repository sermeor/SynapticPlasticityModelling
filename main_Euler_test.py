from scipy.integrate._ivp.common import validate_max_step
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math 
from comp_model import *


#the equations model the differential terms of this model, calculate how much these variables are going to change in each iterations
#find the ks that give the diff eq to 0

#Time array.
t_factor = 3600 # Time factor for graphs.
time = 4*3600/t_factor # Time of simulation depending on t_factor.
sampling_rate = 10*t_factor #number of samples per time factor units.
time_array = np.linspace(0, time, math.floor(time * sampling_rate + 1))
h = time_array[2] - time_array[1] #Step size

## Compartmental models parameters.
weight = 20                          # Mouse weight in g
volume_factor = 5                                                      # ml/kg of body weight.
volume_injection = volume_factor*(weight/1000)                         # in ml.
IP_volume = 2 # in ml.
plasma_volume = 2 # in ml.
brain_volume = 0.41 # in ml.
peripheral_volume = 25 # in ml.

# Volumes in mL.
v0 = IP_volume+volume_injection
v1 = plasma_volume
v2 = brain_volume
v3 = peripheral_volume

#Dose parameters for escitalopram. 
SSRI_dose_factor = 10                     # mg/kg of body weight. 
SSRI_start_time = 0*3600/t_factor           # Starting time of SSRI dose in same units as t_factor.
SSRI_dose = (SSRI_dose_factor*1e6)*(weight/1000) * 0.001 # In ug. 
SSRI_repeat_time = 8*3600/t_factor #Time for repeat of dose. 
SSRI_bioavailability = 0.8

#Molecular weight of escitalopram.
ssri_molecular_weight = 324.392 # g/mol, or ug/umol.


#Dose parameters for FMH. 
FMH_dose_factor = 0 #mg/kg of body weight.
FMH_start_time = 1*3600/t_factor           # Starting time of SSRI dose in same units as t_factor.
FMH_dose = (FMH_dose_factor*1e6)*(weight/1000) * 0.001 # In ug.
FMH_repeat_time = 3600/t_factor # Time for repeat of dose. 
FMH_bioavailability = 0.95

#Molecular weight of FMH.
fmh_molecular_weight = 187.17 # g/mol, or ug/umol.

## Mast cell model of neuroinflammation. 
mc_start_time = 0.5*3600/t_factor #Time to start neuroinflammation effects with mast cells.
mc_switch = 0 #Switch that turns on an off all effects of mast cell presence.

## Basal parameters. 
btrp0 = 96 #Blood tryptophan equilibrium value. 
eht_basal = 0.06 #Steady state basal concentration of serotonin.
gstar_5ht_basal = 0.8561 #Equilibrium concentration of g* serotonin.
gstar_ha_basal =  0.7484 #Equilibrium concentration of g* histamine. 
bht0 = 100 # Blood histidine equilibrium value. 
vht_basal = 63.0457 #Basal vesicular 5ht. 
vha_basal = 136.3639 #Basal vesicular ha.


#Initial conditions
y0 = np.array([95.9766, 0.0994, 0.9006, 20.1618, 1.6094, 0.0373, 63.0383, 280.0048, 0.0603, 1.6824, 113.4099, 0.8660, 1.0112, 0.9791, 0.0027, 0.7114, 1.3245, 0.9874, 0.2666, 1.0203, 0.2297, 0, 0, 0, 0, 0, 3.1074, 136.3639, 241.9217, 1.4378, 2.0126, 99.7316, 249.3265, 311.6581, 0.7114, 1.3245, 0.9874, 0.8660, 1.0112, 0.9791, 354.6656, 177.3328,	350,	150,	3,	140, 0, 0, 0, 0, 1])



#Get solution of the differential equation.

dy = np.zeros(len(y0))
y = y0
y_array = np.array([y])

for t in time_array:
  dy = comp_model(t, y, v2, ssri_molecular_weight, SSRI_start_time, SSRI_repeat_time, SSRI_dose*SSRI_bioavailability, 
  fmh_molecular_weight, FMH_start_time, FMH_repeat_time, FMH_dose*FMH_bioavailability , mc_switch, 
  mc_start_time, btrp0, eht_basal, gstar_5ht_basal, gstar_ha_basal, bht0, 
  vht_basal, vha_basal)

  y = y + dy * h

  
  y_array = np.concatenate((y_array, np.array([y])))




plt.figure()
plt.plot(y_array[:,8])
plt.xlabel('Time (h)')
plt.ylabel('e5HT concentration (nM)')
plt.show()


