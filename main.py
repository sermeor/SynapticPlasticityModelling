from scipy.integrate._ivp.common import validate_max_step
import numpy as np
from numba import njit, jit
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math 
from comp_model import *

#the equations model the differential terms of this model, calculate how much these variables are going to change in each iterations
#find the ks that give the diff eq to 0

#Time array (always in hours).
t_factor = 3600 # Time factor for graphs (1h -> 3600s).
#time = (100/3600000)*3600/t_factor # Time of simulation depending on t_factor.
#sampling_rate = 1*t_factor #number of samples per time factor units.
time = (10000/3600000) #in h
sampling_rate = 1*3600000 #in h-1
time_array = np.linspace(0, time, math.floor(time * sampling_rate + 1))


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
SSRI_dose_factor = 0                     # mg/kg of body weight. 
SSRI_start_time = 0*3600/t_factor           # Starting time of SSRI dose in same units as t_factor.
SSRI_dose = (SSRI_dose_factor*1e6)*(weight/1000) * 0.001 # In ug. 
SSRI_repeat_time = 8*3600/t_factor #Time for repeat of dose. 
SSRI_bioavailability = 0.8

#Molecular weight of escitalopram.
ssri_molecular_weight = 324.392 # g/mol, or ug/umol.

#Dose parameters for FMH. 
FMH_dose_factor = 0 #mg/kg of body weight.
FMH_start_time = 0*3600/t_factor           # Starting time of SSRI dose in same units as t_factor.
FMH_dose = (FMH_dose_factor*1e6)*(weight/1000) * 0.001 # In ug.
FMH_repeat_time = 3600/t_factor # Time for repeat of dose. 
FMH_bioavailability = 0.95

#Molecular weight of FMH.
fmh_molecular_weight = 187.17 # g/mol, or ug/umol.

#Dose parameters for ketamine. 
ket_dose_factor = 0                     # mg/kg of body weight. 
ket_start_time = 0*3600/t_factor           # Starting time of ketamine dose in same units as t_factor.
ket_dose = (ket_dose_factor*1e6)*(weight/1000) * 0.001 # In ug. 
ket_repeat_time = 8*3600/t_factor #Time for repeat of dose. 
ket_bioavailability = 0.8

#Molecular weight of escitalopram.
ket_molecular_weight = 237.725 #g/mol, or ug/umol.
norket_molecular_weight = 260.16 #g/mol, or ug/umol.

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



##Neural network model. 
#Constant parameters
N = 5  #Number of neurons.
NE = int(0.8 * N)  #Number of excitatory neurons.

random_seed = 25  #Seed of the pseudo-random number generator.
set_seed(random_seed)  #Set seed in compiled code.
np.random.seed(random_seed)  #Set seed in non-compiled code.

#Initial conditions with their original shapes.
Vm = -60 * np.ones(N)  #Membrane potential.
m = 0.05 * np.ones(N)  #Activation gating variable for the voltage-gated sodium (Na+) channels.
h = 0.6 * np.ones(N)  #Activation gating variable for the voltage-gated potassium (K+) channels.
n = 0.32 * np.ones(N)  #Inactivation gating variable for the Na+ channels.
c = 0 * np.ones(N)  #Gatting variable for VGCC.
Ca_0 = 0.1 * np.ones(N)  #Internal calcium.
N_0 = np.zeros(N)  #Neurotransmitter.
g_fast_0 = np.zeros(N)  #Fast conductances.
g_slow_0 = np.zeros(N)  #Membrane potential. #Slow conductances.
act = 0.5 * np.ones(N)  #Activity state.

C = np.random.rand(N, N).flatten()  #Connectivity matrix.

ynn0 = np.concatenate((Vm, m, h, n, c, Ca_0, N_0, g_fast_0, g_slow_0, act, C), axis=0)  #Flatten initial conditions.





#Initial conditions
y0_ser_model = np.array([95.9766, 0.0994, 0.9006, 20.1618, 1.6094, 0.0373, 63.0383, 280.0048, 0.0603, 1.6824, 113.4099, 0.8660, 1.0112, 0.9791, 0.0027, 0.7114, 1.3245, 0.9874, 0.2666, 1.0203, 0.2297, 0, 0, 0, 0, 0, 3.1074, 136.3639, 241.9217, 1.4378, 2.0126, 99.7316, 249.3265, 311.6581, 0.7114, 1.3245, 0.9874, 0.8660, 1.0112, 0.9791, 354.6656, 177.3328,	350,	150,	3,	140, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

y0 = np.concatenate((y0_ser_model, ynn0), axis=0)

arguments = (v2, ssri_molecular_weight, SSRI_start_time, SSRI_repeat_time, SSRI_dose*SSRI_bioavailability, fmh_molecular_weight, FMH_start_time, FMH_repeat_time, FMH_dose*FMH_bioavailability, ket_start_time, ket_repeat_time, ket_dose*ket_bioavailability, ket_molecular_weight, norket_molecular_weight, mc_switch, mc_start_time, btrp0, eht_basal, gstar_5ht_basal, gstar_ha_basal, bht0, vht_basal, vha_basal, N, NE)
 
#Get solution of the differential equation.

sol = solve_ivp(comp_model, t_span = (time_array[0], time_array[-1]), t_eval = time_array, y0 = y0, method = 'RK45', args = arguments)


plt.figure(1)
plt.plot(sol.y[59, :])
plt.xlabel('Time (h)')
plt.ylabel('Vm')
plt.show()



