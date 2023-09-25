import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math 

def ketamine_inj(t, t_start, t_repeat, q):
  inj_time = 1/3600
  if t > t_start:
    n_stim = int((t - t_start)//t_repeat)
    if (t > (t_start + n_stim * t_repeat)) and (t <= (t_start + inj_time + n_stim * t_repeat)):
      f = q/inj_time
    else:
      f = 0
  else:
    f = 0
  return f

##Function of occupacy ratio of NMDA receptors depending on ketamine and norketamine concentration (uM)
#This function assumes that k and nk don't inhibit each other, and that glutamate concentration has no effect on binding rate of k and nk, since their binding stregth is much greater. 
def inhib_NMDA(k, nk):
  #Hill equation (MM non-competitive inhibition)
  #Not affected by glutamate concentration.
  n_k = 1.4 #Hill number ketamine. 
  n_nk = 1.2 #Hill number nor-ketamine.
  Ki_k = 2 #Ketamine concentration for half occupacy (uM)
  Ki_nk = 17 #Norketamine concentration for half occupacy (uM)
  
  f = 1/(1 + (Ki_k/k)**n_k) + 1/(1 + (Ki_nk/nk)**n_nk)

  return f

# function that returns dz/dt
def comp_model(t, y, ketamine_start_time, ketamine_repeat_time, ketamine_q_inj):
  #Constant rates in min-1, from literature.
  
  k01 = 0.75*1.5625
  k10 = 1.4*82.5
  k12 = 0.75*28.5
  k21 = 5.25
  k13 = 100
  k31 = 5


  k10_nk = 0.5*82.5
  k12_nk = 0.5*28.5
  k21_nk = 0.75*5.25
  k13_nk = 0.8*100
  k31_nk = 5
	
  q1 = 0.5*162.5 #ketamine -> norketamine in blood (h^-1). 
  q2 = 0.05 #ketamine -> norketamine in brain (h^-1). 
  q3 = 0.5*25 #ketamine -> norketamine in periphery (liver)  #(h^-1).

  # y[0] = Peritoneum concentration of ketamine in ug.
  # y[1] = Blood concentration of ketamine in ug.
  # y[2] = Blood concentration of norketamine in ug.
  # y[3] = Brain concentration of ketamine in ug.
  # y[4] = Brain concentration of norketamine in ug.
  # y[5] = Periphery concentration of ketamine in ug.
  # y[6] = Periphery concentration of norketamine in ug. 

  # Parameters.
  protein_binding_k = 0.60
  protein_binding_nk = 0.5
  protein_brain_binding = 0.15
  
  dy = np.zeros(7)
  #Equations
  dy[0] = ketamine_inj(t, ketamine_start_time, ketamine_repeat_time, ketamine_q_inj) - k01*(y[0])
  
  dy[1] = k01*(y[0]) - (k10 + k12+ k13)*(y[1]*(1-protein_binding_k)) + k21*(y[3]*(1-protein_brain_binding)) + k31*(y[5]) - q1 * y[1]*(1-protein_binding_k)

  dy[2] = q1 * y[1] * (1-protein_binding_k) - (k10_nk + k12_nk + k13_nk)*(y[2]*(1-protein_binding_nk)) + k21_nk*(y[4]*(1-protein_brain_binding)) + k31_nk*(y[6])
  
  dy[3] = k12*(y[1]*(1-protein_binding_k)) - k21*(y[3]*(1-protein_brain_binding)) - q2 * y[3]*(1-protein_brain_binding)

  dy[4] = q2 * y[3]*(1-protein_brain_binding) + k12_nk*y[2]*(1-protein_binding_nk) - k21_nk*(y[4]*(1-protein_brain_binding))
  
  dy[5] = k13*(y[1]*(1-protein_binding_k)) - k31*(y[5]) - q3 * y[5]

  dy[6] = q3 * y[5] + k13_nk*(y[2]*(1-protein_binding_nk)) - k31_nk*(y[6])
  
  return dy





#Time array.
t_factor = 3600 # Time factor for graphs.
time = 25*3600/t_factor # Time of simulation depending on t_factor.
sampling_rate =1*t_factor #number of samples per time factor units.
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

#Dose parameters for ketamine. 
ketamine_dose_factor = 10                     # mg/kg of body weight. 
ketamine_start_time = 0*3600/t_factor           # Starting time of ketamine dose in same units as t_factor.
ketamine_dose = (ketamine_dose_factor*1e6)*(weight/1000) * 0.001 # In ug. 
ketamine_repeat_time = 80*3600/t_factor #Time for repeat of dose. 
ketamine_bioavailability = 0.8

#Molecular weight of escitalopram.
ketamine_molecular_weight = 237.725 #g/mol, or ug/umol.
norket_molecular_weight = 260.16 #g/mol, or ug/umol.



#Initial conditions
y0 = [0, 0, 0, 0, 0, 0, 0]

arguments = (ketamine_start_time, ketamine_repeat_time, ketamine_dose*ketamine_bioavailability)
 
#Get solution of the differential equation.

sol = solve_ivp(comp_model, t_span = (time_array[0], time_array[-1]), t_eval = time_array, y0 = y0, method = 'RK45', args = arguments)



plt.figure()
plt.subplot(4,1,1)
plt.plot(time_array, sol.y[0, :])
plt.ylabel('C0 (ug)')
plt.subplot(4,1,2)
plt.plot(time_array, sol.y[1, :])
plt.plot(time_array, sol.y[2, :])
plt.ylabel('C1 (ug)')
plt.subplot(4,1,3)
plt.plot(time_array, sol.y[3, :])
plt.plot(time_array, sol.y[4, :])
plt.ylabel('C2 (ug)')
plt.subplot(4,1,4)
plt.plot(time_array, sol.y[5, :])
plt.plot(time_array, sol.y[6, :])
plt.ylabel('C3 (ug)')
plt.xlabel('time (h)')
plt.show()
