import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import math


## Mathematical model of synaptogenesis. 

##Function of rate constant increase of m.
def alpha_m(Vm):
   return 0.1*(Vm + 40.0)/(1.0 - np.exp( - (Vm + 40.0)/10.0))

##Function of rate constant decrease of m.
def beta_m(Vm):
   return 4.0*np.exp( - (Vm + 65.0)/18.0) 

##Function of rate constant increase of h.
def alpha_h(Vm):
   return 0.07 * np.exp( - (Vm + 65.0)/20.0)

##Function of rate constant decrease of h.
def beta_h(Vm):
   return 1.0/(1.0 + np.exp( - (Vm + 35.0)/10.0))
   
##Function of rate constant increase of n. 
def alpha_n(Vm):
  return 0.01 * (Vm + 55.0)/(1.0 - np.exp( - (Vm + 55.0)/10.0))

##Function of rate constant decrease of n. 
def beta_n(Vm):
  return 0.125 * np.exp( - (Vm + 65)/80.0)

## Function of sodium channel current (mA/cm^2). 
def I_Na(m, h, Vm):
  g_Na = 120.0  # maximum sodium conductance (mS/cm^2)
  E_Na = 50.0  # sodium reversal potential (mV)
  return g_Na*(m**3)*h*(Vm - E_Na)

##Function of potassium channels current (mA/cm^2).
def I_K(n, Vm):
  E_K = -77.0  # potassium reversal potential (mV)
  g_K = 36.0  # maximum potassium conductance (mS/cm^2)
  return g_K*(n**4)*(Vm - E_K)

##Function of leakage current (mA/cm^2).
def I_L(Vm):
  g_L = 0.3  # maximum leak conductance (mS/cm^2)
  E_L = -54.4  # leak reversal potential (mV)
  return g_L*(Vm - E_L)
  
##Function of AMPA current (mA/cm^2). 
def  I_AMPA(g_AMPA, Vm, E_AMPA):
  return g_AMPA*(Vm - E_AMPA)

##Function of NMDA current (mA/cm^2). 
def  I_NMDA(g_NMDA, Vm, E_NMDA):
  return g_NMDA*(Vm - E_NMDA)/(1.0 + 0.28*np.exp(-0.062*Vm))
  
##Function of GABA A current (mA/cm^2).
def I_GABA_A(g_GABA_A, Vm, E_GABA_A):
  return g_GABA_A * (Vm - E_GABA_A)
  
##Function of GABA B current (mA/cm^2).
def I_GABA_B(g_GABA_B, Vm, E_GABA_B):
  return g_GABA_B*(Vm - E_GABA_B)/(1.0 + np.exp(-(Vm + 80.0)/25.0))



## ODE model
def comp_model(t, y, N, NE, theta, phi, beta, th):
  # Parameters of HH model
  C_m = 1.0  # membrane capacitance (uF/cm^2)
  # Parameters for synaptic currents
  E_AMPA = 0.0 #Reversal potential for AMPA channels (mV)
  E_NMDA = 0.0 #Reversal potential for NMDA channels (mV)
  E_GABA_A = -70.0 #Reversal potential for GABA A channels (mV)
  E_GABA_B = -90.0 #Reversal potential for GABA B channels (mV)
  
  F = 9.649 * 10**4/(10**6) # Faraday Constant in coulomb/umol
  V_n = 1 # Volume of neuron shell (L)  
  alpha = 0.1 # rate determining duration taken into account when averaging membrane potential
  
  #[NEED CHANGING, DEPENDS ON CONNECTIVITY]
  # g_AMPA = k1*sum(y[6][i,:NE] * y[0][:NE])
  # g_GABA_A = k2*sum(y[6][i,NE:] * y[0][NE:])


  g_AMPA = 0 #Conducance of AMPA channels (mS/cm^2). 
  g_NMDA = 0 #Conductance of NMDA channels (mS/cm^2).
  g_GABA_A = 0 #Conductance of GABA A channels (mS/cm^2).
  g_GABA_B = 0 #Conducance of GABA B channels (mS/cm^2),

  #Structure flatten y into list of arrays. 
  y = [y[:N], y[N:2*N], y[2*N:3*N], y[3*N:4*N], y[4*N:5*N], y[5*N:6*N], y[6*N:7*N], y[7*N:].reshape(N, N)]

  #Initialize differential list. 
  dy = [0, 0, 0, 0, 0, 0, 0, 0]

  #Variable definitions. 
  #y[0] = state of network over time.  
  #y[1] = probability pre-synaptic neuron having an effect on post-synaptic neuron. 
  #y[2] = membrane potential of neurons. 
  #y[3] = m
  #y[4] = h
  #y[5] = n
  #y[6] = internal calcium concentration (uM)
  #y[7] = connectivity matrix.
  #y[8] = online average of membrane potential  


  # Synaptic currents
 



  #Differential equations
  
  dy[0] = [1 if i > 0.5 else 0 for i in y[1]] - y[0] #np.array([np.random.binomial(n = 1, p = i) for i in y[1]]) - y[0]  #Discrete, does not depend on th
  dy[1] = (1/(1 + np.exp( - (y[2] - theta)/beta)) - y[1])/th
  dy[2] = (- I_Na(y[3], y[4], y[2]) - I_K(y[5], y[2]) - I_L(y[2]) - I_AMPA(g_AMPA, y[2], E_AMPA) - I_NMDA(g_NMDA, y[2], E_NMDA) - I_GABA_A(g_GABA_A, y[2], E_GABA_A) - I_GABA_B(g_GABA_B, y[2], E_GABA_B))/C_m
  dy[3] = alpha_m(y[2])*(1.0 - y[3]) - beta_m(y[2])*y[3]
  dy[4] = alpha_h(y[2])*(1.0 - y[4]) - beta_h(y[2])*y[4]
  dy[5] = alpha_n(y[2])*(1.0 - y[5]) - beta_n(y[2])*y[5]

  p_Ca = 10.6/(10.6+1+1) # relative permeability of NMDA receptor to Calcium
  #dy[6] = np.zeros(N) #nmda_ca_influx(I_NMDA) - ca_pump()

  dy[6] = (p_Ca * I_NMDA(g_NMDA, y[2], E_NMDA))/(2*F*V_n) # fraction of ion flux attributed to calcium
  
  dy[7] = np.zeros([N, N]).flatten()
  #Connectivity matrix to dynamically change based on the level of pre and post synaptic inhibitory and excitatory currents. 
  #Look at original paper. 

  dy[8] = y[8] + alpha * (y[8] - y[2])


  #flatten dy
  dy = np.concatenate(dy).flatten()


  return dy


#Constant parameters
N = 30 #Number of neurons. 
NE = int(0.8*N) #Number of excitatory neurons.
theta = -55 #Membrane potential threshold of neuronal activation.
phi = 8 #Relative weight of input from inhibitory neurons respect to excitatory. 
beta = 2 #Noise level of threshold function. 

#Time array.
t_factor = 1 # Time factor for graphs.
time = 100/t_factor # Time of simulation depending on t_factor.
sampling_rate = 10*t_factor #number of samples per time factor units.
time_array = np.linspace(0, time, math.floor(time * sampling_rate + 1))
th = time_array[2] - time_array[1] #Time interval between samples.

#Initial conditions with their original shapes. 
z = np.zeros(N) #State of network over time.
prob = np.zeros(N) #Probability of a neuron to be active. 
Vm = -65.0 * np.ones(N) #Membrane potential. 
m = 0.05 * np.ones(N) #Activation gating variable for the voltage-gated sodium (Na+) channels.
h = 0.6 * np.ones(N) #Activation gating variable for the voltage-gated potassium (K+) channels.
n = 0.32 * np.ones(N) #Inactivation gating variable for the Na+ channels.

Ca_0 = np.zeros(N) #Internal calcium. 

C = np.zeros([N, N]).flatten() #Connectivity matrix. 

y0 = np.concatenate((z, prob, Vm, m, h, n, Ca_0, C), axis=0) #Flatten initial conditions.

arguments = (N, NE, theta, phi, beta, th)

#Get solution of the differential equation.

sol = solve_ivp(comp_model, t_span = (time_array[0], time_array[-1]), t_eval = time_array, y0 = y0, method = 'DOP853', args = arguments)


# Get results
y = [sol.y[:N, :], sol.y[N:2*N, :], sol.y[2*N:3*N, :], sol.y[3*N:4*N, :], sol.y[4*N:5*N, :], sol.y[5*N:6*N, :], sol.y[6*N:7*N, :], sol.y[7*N:, :].reshape(N, N, sol.t.shape[0])]

plt.figure()
plt.subplot(3,1,1)
plt.plot(y[2][0,:])
plt.subplot(3,1,2)    
plt.plot(I_AMPA(0.1, y[2][0,:], 0))
plt.subplot(3,1,3)  
plt.plot(I_NMDA(0.01, y[2][0,:], 0))
plt.show()

