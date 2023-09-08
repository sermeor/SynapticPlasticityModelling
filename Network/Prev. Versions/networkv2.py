
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
def  I_AMPA(g_AMPA, Vm):
  E_AMPA = 0.0 #Reversal potential for AMPA channels (mV)
  return g_AMPA*(Vm - E_AMPA)

##Function of magnesium block of NMDA dependent on voltage (mV).
def MgB(Vm):
  Mg0 = 2 #mM
  return 1/(1 + (Mg0 * np.exp(-0.062*Vm))/3.57)


##Function of NMDA channel sodium current density (mA/cm^2)
def I_NMDA_Na(g_NMDA, Vm):
  P_Na = 1 #Permeability ratio to sodium. 
  c = 0.1 #Conversor A/m^2 -> mA/cm^2 
  P_NMDA = 10*10**(-9) #m/s
  F = 96485 #C/mol 
  R = 8.314 #J/K*mol
  T = 308.15 #K
  Nai = 18 #mM
  Nao = 140 #mM
  return g_NMDA * c * P_NMDA * P_Na * MgB(Vm) * ((Vm/1000 * F**2)/(R*T))*((Nai - Nao * np.exp(-((Vm/1000 * F)/(R*T))))/(1 - np.exp(-((Vm/1000 * F)/(R*T)))))

##Function of NMDA channel potassium current density (mA/cm^2)
def I_NMDA_K(g_NMDA, Vm):
  P_K = 1 #Permeability ratio to potassium. 
  c = 0.1 #Conversor A/m^2 -> mA/cm^2 
  P_NMDA = 10*10**(-9) #m/s
  F = 96485 #C/mol 
  R = 8.314 #J/K*mol
  T = 308.15 #K
  Ki = 140 #mM
  Ko = 5 #mM
  return g_NMDA * c * P_NMDA * P_K * MgB(Vm) * ((Vm/1000 * F**2)/(R*T))*((Ki - Ko * np.exp(-((Vm/1000 * F)/(R*T))))/(1 - np.exp(-((Vm/1000 * F)/(R*T)))))

  
##Function of NMDA channel calcium current density (mA/cm^2).
def I_NMDA_Ca(g_NMDA, Vm):
  P_Ca = 10.6 #Permeability ratio to calcium. 
  c = 0.1 #Conversor A/m^2 -> mA/cm^2 
  P_NMDA = 10*10**(-9) #m/s
  F = 96485 #C/mol 
  R = 8.314 #J/K*mol
  T = 308.15 #K
  Cai = 0.0001 #mM
  Cao = 2 #mM
  return g_NMDA * c * P_NMDA * P_Ca * MgB(Vm) * ((4*Vm/1000 * F**2)/(R*T))*((Cai - Cao * np.exp(-((2*Vm/1000 * F)/(R*T))))/(1 - np.exp(-((2*Vm/1000 * F)/(R*T)))))


##Function of total NMDA channel current density (mA/cm^2).
def I_NMDA(g_NMDA, Vm):
  return I_NMDA_Na(g_NMDA, Vm) + I_NMDA_K(g_NMDA, Vm) + I_NMDA_Ca(g_NMDA, Vm)


##Function of inward calcium rate (uM/ms).
def inward_Ca(g_NMDA, Vm):
  F = 96485 # Faraday Constant (mA*ms/umol). 
  d = 8.4e-6 #Distance of membrane shell where calcium ions enter (cm).  
  c = 1000 #conversor umol/(cm^3 * ms) to uM/ms. 
  return - c * I_NMDA_Ca(g_NMDA, Vm)/(2*F*d)

##Function of outward calcium rate (uM/ms).
def outward_Ca(Cai):
  Cai_eq = 0
  c = 0.1 #Rate of calcium pump buffering (ms^-1).
  return + c * (Cai - Cai_eq)



  
##Function of GABA A current (mA/cm^2).
def I_GABA_A(g_GABA_A, Vm, E_GABA_A):
  return g_GABA_A * (Vm - E_GABA_A)
  
##Function of GABA B current (mA/cm^2).
def I_GABA_B(g_GABA_B, Vm, E_GABA_B):
  return g_GABA_B*(Vm - E_GABA_B)/(1.0 + np.exp(-(Vm + 80.0)/25.0))


def dummy_NMDA(t):
  if t>5 and t<7:
    a = 0.5
  else:
    a = 0
  return a

def g_AMPA(spikes, t, t_curr, NE):
  tau_rise = 5 #ms
  tau_fast = 5 #ms
  tau_slow = 5 #ms
  g_bar = 40 #pS
  a = 1 # constant

  spike_times = np.nonzero(spikes)
  
  g = 0 # potentially needs better value for equilibrium state

  for i in spike_times:
    rise = 1 - np.exp(-(t_curr-t[i])/tau_rise)
    fast = a * (1 - np.exp(-(t_curr-t[i])/tau_fast))
    slow = (1-a) * (1 - np.exp(-(t_curr-t[i])/tau_slow))

    g += g_bar * rise * (fast + slow) * np.heaviside(t_curr - t[i], 1)
  
  return g

def g_NMDA(spikes, t, t_curr):
  tau_rise = 5 #ms
  tau_decay = 50 #ms
  g_bar = 1.5 * 10**3 #pS
  a = 1 # constant
  beta = 1/3.57 #constant 1/mM
  alpha = 0.062 #constant mV
  u = -65
  c_mg = 1.2 # mM

  spike_times = np.nonzero(spikes)
  
  g = 0 # potentially needs better value for equilibrium state

  for i in spike_times:
    rise = 1 - np.exp(-(t_curr-t[i])/tau_rise)
    decay = np.exp(-(t_curr-t[i])/tau_decay)
    g_infinity = 1/(1 + beta * np.exp(-alpha * u * c_mg))

    g += g_bar * rise * decay * g_infinity * np.heaviside(t_curr-t[i], 1)
  
  return g

# returns index in flattened version of array
# desired index i, #neurons N, data y
def index(i, N, y):
  new_i = [x for x in range(i*N, (i+1)*N)]
  
  return new_i
## ODE model
def comp_model(t, y, N, NE, theta, phi, beta, th):
  # Parameters of HH model
  C_m = 1.0  # membrane capacitance (uF/cm^2)

  # Parameters for synaptic currents
  E_NMDA = 0.0 #Reversal potential for NMDA channels (mV).
  E_GABA_A = -70.0 #Reversal potential for GABA A channels (mV).
  E_GABA_B = -90.0 #Reversal potential for GABA B channels (mV).
  

  alpha = 0.1 # rate determining duration taken into account when averaging membrane potential
  
  #[NEED CHANGING, DEPENDS ON CONNECTIVITY AND TIME]
  # g_AMPA = k1*sum(y[6][i,:NE] * y[0][:NE])
  # g_GABA_A = k2*sum(y[6][i,NE:] * y[0][NE:])


  #Structure flatten y into list of arrays. 
  y = [y[:N], y[N:2*N], y[2*N:3*N], y[3*N:4*N], y[4*N:5*N], y[5*N:6*N], y[6*N:7*N], y[7*N:].reshape(N, N)]

  #Initialize differential list. 
  dy = [0, 0, 0, 0, 0, 0, 0, 0]
  
  g_AMPA = dummy_NMDA(t) #Factor of conductance of AMPA channels. 
  g_NMDA = dummy_NMDA(t) #Conductance of NMDA channels.
  g_GABA_A = 0 #Conductance of GABA A channels.
  g_GABA_B = 0 #Conducance of GABA B channels.

  #Variable definitions. 
  #y[0] = state of network over time.
  #y[1] = probability of pre-synaptic neuron firing.
  #y[2] = membrane potential of neurons. 
  #y[3] = m
  #y[4] = h
  #y[5] = n
  #y[6] = internal calcium concentration (uM)
  #y[7] = connectivity matrix.
  #y[8] = online average of membrane potential  

  state_network = np.array([np.random.binomial(n = 1, p = i) for i in y[1]])


  #Differential equations
  dy[0] = [1 if i > 0.5 else 0 for i in y[1]] - y[0] #np.array([np.random.binomial(n = 1, p = i) for i in y[1]]) - y[0]  #Discrete, does not depend on th
  dy[1] = (1/(1 + np.exp( - (y[2] - theta)/beta)) - y[1])/th
  dy[2] = (- I_Na(y[3], y[4], y[2]) - I_K(y[5], y[2]) - I_L(y[2]) - I_AMPA(g_AMPA, y[2]) - I_NMDA(g_NMDA, y[2]) - I_GABA_A(g_GABA_A, y[2], E_GABA_A) - I_GABA_B(g_GABA_B, y[2], E_GABA_B))/C_m
  dy[3] = alpha_m(y[2])*(1.0 - y[3]) - beta_m(y[2])*y[3]
  dy[4] = alpha_h(y[2])*(1.0 - y[4]) - beta_h(y[2])*y[4]
  dy[5] = alpha_n(y[2])*(1.0 - y[5]) - beta_n(y[2])*y[5]
  dy[6] = inward_Ca(g_NMDA, y[2]) - outward_Ca(y[6])




  dy[7] = np.zeros([N, N]).flatten()
  #Connectivity matrix to dynamically change based on the level of pre and post synaptic inhibitory and excitatory currents. 
  #Look at original paper. 

  #dy[8] = y[8] + alpha * (y[8] - y[2])


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
time = 10/t_factor # Time of simulation depending on t_factor.
sampling_rate = 100*t_factor #number of samples per time factor units.
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
a = [dummy_NMDA(t) for t in time_array]
curr = [I_NMDA(a[i], y[2][0,:][i]) for i in range(0, len(time_array))]
plt.plot(curr)
plt.subplot(3,1,3)    
plt.plot(y[6][0, :])
plt.show()