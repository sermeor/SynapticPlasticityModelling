import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import math
import time


%matplotlib inline
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
  V_lim = 100 #mV
  a1 = g_NMDA * c * P_NMDA * P_Na * MgB(Vm) * ((Vm/1000 * F**2)/(R*T))
  a2 = np.array([Nai if Vmi > V_lim else Nao if Vmi < -V_lim else ((Nai - Nao * np.exp(-((Vmi/1000 * F)/(R*T))))/(1 - np.exp(-((Vmi/1000 * F)/(R*T))))) for Vmi in Vm])

  I = a1*a2

  return I

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
  V_lim = 100 #mV
  a1 = g_NMDA * c * P_NMDA * P_K * MgB(Vm) * ((Vm/1000 * F**2)/(R*T))
  a2 = np.array([Ki if Vmi > V_lim else Ko if Vmi < -V_lim else ((Ki - Ko * np.exp(-((Vmi/1000 * F)/(R*T))))/(1 - np.exp(-((Vmi/1000 * F)/(R*T))))) for Vmi in Vm])
  I = a1*a2

  return I


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
  V_lim = 100 #mV
  a1 =  g_NMDA * c * P_NMDA * P_Ca * MgB(Vm) * ((4*Vm/1000 * F**2)/(R*T))
  a2 = np.array([Cai if Vmi > V_lim else Cao if Vmi < -V_lim else ((Cai - Cao * np.exp(-((2*Vmi/1000 * F)/(R*T))))/(1 - np.exp(-((2*Vmi/1000 * F)/(R*T))))) for Vmi in Vm])

  I = a1*a2

  return I


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
def I_GABA_A(g_GABA_A, Vm):
  E_GABA_A = -70.0 #Reversal potential for GABA A channels (mV).
  return g_GABA_A * (Vm - E_GABA_A)

##Function of GABA B current (mA/cm^2).
def I_GABA_B(g_GABA_B, Vm):
  E_GABA_B = -95.0 #Reversal potential for GABA B channels (mV).
  return g_GABA_B*(Vm - E_GABA_B)/(1.0 + np.exp(-(Vm + 80.0)/25.0))


##Function that draws a Bernoulli sample from the probability of neuron firing.
#Output is 1 (active) or 0 (inactive). Seed is fixed to have the same results.
def spike_boolean(Vm):
  Vth = 0
  return np.array(Vm >= Vth, dtype = int)

#Function that updates the connectivity matrix depending on activity of neuron (pre-synaptic connectivity). 


def connectivity_update(C, act, N, NE):
    #Postsynaptic update.
    sigma = 0.5 #Midpoint activity, below this value, the activity of neuron is low, and above the value is high.
    w_ex = -0.001 #Weight of update on excitatory plates.
    w_inh = 0.001 #Weight of update on inhibitory plates.
    post_delta_C_row = (act - sigma)
    post_delta_C = np.full((N, N), post_delta_C_row)  # Create an N x N matrix with delta_C_row as values
    post_delta_C[:NE] *= w_ex  # Multiply the first NE rows by w_ex
    post_delta_C[NE:] *= w_inh  # Multiply the rest of the rows by w_inh

    #Presynaptic update.
    w = 0.001 #Weight of update. 
    pre_delta_C_row = w * (act - sigma) #ΔC for all neurons (1, 2,... N).
    pre_delta_C = np.tile(pre_delta_C_row, (N, 1)).T #Delta matrix. 

    #Sum both effects. 
    delta_C = post_delta_C + pre_delta_C

    #Check it is in [0, 1]
    #temp = C + delta_C
    #invalid_indices = np.logical_or(np.logical_and(temp < 0, delta_C < 0), np.logical_and(temp > 0, delta_C > 0))
    #delta_C = np.where(invalid_indices, 0, delta_C)

    return delta_C





## ODE model
def comp_model(t, y, N, NE, th):
  #Structure flatten y into list of arrays.
  y = [y[:N], y[N:2*N], y[2*N:3*N], y[3*N:4*N], y[4*N:5*N], y[5*N:6*N], y[6*N:7*N], y[7*N:8*N], y[8*N:9*N], y[9*N:].reshape(N, N)]

  # Parameters of HH model
  C_m = 1.0  # membrane capacitance (uF/cm^2)

  #Constants
  a1 = 0.25 #Weight of excitatory AMPA inputs.
  a2 = 0.25 #Weight of inhibitory GABA A inputs.
  a3 = 0.25 #Weight of excitatory NMDA inputs.
  a4 = 0.25 #Weight of inhibitory GABA B inputs.


  a5 = 1 #Rate of production of neurotransmitter per spike (ms-1).
  a6 = 0.25 #Rate of reuptake of neurotransmitter (ms-1).

  a7 = 0.05 #Rate of increase of excitatory w_fast (AMPA) from neurotransmitter presence (ms-1).
  a8 = 0.1 #Rate of decrease of excitatory w_fast (AMPA) (ms-1).

  a9 = 0.03 #Rate of increase of inhibitory w_fast (GABA A) from neurotransmitter presence (ms-1).
  a10 = 0.1 #Rate of decrease of inhibitory w_fast (GABA A) (ms-1).

  a11 = 0.025 #Rate of increase of excitatory w_slow (NMDA) from neurotransmitter presence (ms-1).
  a12 = 0.020 #Rate of decrease of excitatory w_slow (NMDA) (ms-1).

  a13 = 0.001 #Rate of increase of inhibitory w_slow (GABA B) from neurotransmitter presence (ms-1).
  a14 = 0.001 #Rate of decrease of inhibitory w_slow(GABA B) from neurotransmitter presence (ms-1).

  a15 = 0.01 #Rate of increase of neuron activity from individual spike (ms-1).
  a16 = 0.0001 #Rate of decrease of neuron activity if neuron does not spike (ms-1). 


  spike = spike_boolean(y[0]) #Discrete spikes.

  # Define Poisson input parameters
  rate = 10.0  # firing rate (Hz)
  w1 = 0.75  # Excitatory noise synaptic weight
  w2 = 0.2 #Inhibitory noise synaptic weight
  I_noise_e = w1 * np.random.poisson(rate*th, N)  #Poisson excitatory input noise.
  I_noise_i = w2 * np.random.poisson(rate*th, N)  #Poisson inhibitory input noise.


  g_AMPA = np.array([a1 * sum(y[9][:NE, i] * y[6][:NE]) for i in range(0, N)])/NE # Conductance factor of AMPA channels.
  g_GABA_A = np.array([a2 * sum(y[9][NE:, i] * y[6][NE:]) for i in range(0, N)])/(N-NE) # Conductance factor of GABA A channels.
  g_NMDA = np.array([a3 * sum(y[9][:NE, i] * y[7][:NE]) for i in range(0, N)])/NE # Conductance factor of NMDA channels.
  g_GABA_B = np.array([a4 * sum(y[9][NE:, i] * y[7][NE:]) for i in range(0, N)])/(N-NE) # Conductance factor of GABA B channels.


  #Initialize differential list.
  dy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

  #Variable in ODE.
  #y[0] = Vm, membrane potential of neurons.
  #y[1] = m, activation gating variable for the voltage-gated sodium (Na+) channels.
  #y[2] = h, activation gating variable for the voltage-gated potassium (K+) channels.
  #y[3] = n, Inactivation gating variable for the Na+ channels.
  #y[4] = Cai, internal calcium concentration (uM)
  #y[5] = N, Neurotransmitter activity in synapse.
  #y[6] = w_fast, synaptic weights of activation of fast receptors (AMPA, GABA A).
  #y[7] = w_slow, synaptic weights of activation of slow receptors (NMDA, GABA B).
  #y[8] = Activity state of neuron (low/high state).
  #y[9] = connectivity matrix.

  #Differential equations
  dy[0] = (- I_Na(y[1], y[2], y[0]) - I_K(y[3], y[0]) - I_L(y[0]) - I_AMPA(g_AMPA, y[0]) - I_NMDA(g_NMDA, y[0]) - I_GABA_A(g_GABA_A, y[0]) - I_GABA_B(g_GABA_B, y[0]) + I_noise_e - I_noise_i)/C_m
  dy[1] = alpha_m(y[0])*(1.0 - y[1]) - beta_m(y[0])*y[1]
  dy[2] = alpha_h(y[0])*(1.0 - y[2]) - beta_h(y[0])*y[2]
  dy[3] = alpha_n(y[0])*(1.0 - y[3]) - beta_n(y[0])*y[3]
  dy[4] = inward_Ca(g_NMDA, y[0]) - outward_Ca(y[4])

  dy[5] = a5 * spike - a6 * y[5]


  dy[6] = np.array([a7 * y[5][i] - a8 * y[6][i] if i<NE else a9 * y[5][i] - a10 * y[6][i] for i in range(0, N)])

  dy[7] =  np.array([a11 * y[5][i] - a12 * y[7][i] if i<NE else a13 * y[5][i] - a14 * y[7][i] for i in range(0, N)])

  dy[8] =  a15 * spike *(1 - y[8]) - a16 * y[8]

  dy[9] = connectivity_update(y[9], y[8], N, NE).flatten()

  #flatten dy
  dy = np.concatenate(dy).flatten()

  return dy






#Constant parameters
N = 25 #Number of neurons.
NE = int(0.8*N) #Number of excitatory neurons.

random_seed = 25 #Seed of the pseudo-random number generator.
np.random.seed(random_seed)

#Time array.
t_factor = 1 # Time factor for graphs.
time = 500/t_factor # Time of simulation depending on t_factor.
sampling_rate = 1*t_factor #number of samples per time factor units.
time_array = np.linspace(0, time, math.floor(time * sampling_rate + 1))
th = time_array[1] - time_array[0] #Time interval between samples.

#Initial conditions with their original shapes.
Vm = np.random.uniform(low=-75, high=50, size=(N,)) #Membrane potential.
m = 0.05 * np.ones(N) #Activation gating variable for the voltage-gated sodium (Na+) channels.
h = 0.6 * np.ones(N) #Activation gating variable for the voltage-gated potassium (K+) channels.
n = 0.32 * np.ones(N) #Inactivation gating variable for the Na+ channels.
Ca_0 = np.zeros(N) #Internal calcium.
N_0 = np.zeros(N)  #Neurotransmitter.
g_fast_0 = np.zeros(N) #Fast conductances.
g_slow_0 = np.zeros(N) #Membrane potential. #Slow conductances.
act = 0.5*np.ones(N) #Activity state. 
C = np.random.rand(N, N).flatten() #Connectivity matrix.

y0 = np.concatenate((Vm, m, h, n, Ca_0, N_0, g_fast_0, g_slow_0, act, C), axis = 0) #Flatten initial conditions.
arguments = (N, NE,  th)


#Get solution of the differential equation.
sol = solve_ivp(comp_model, t_span = (time_array[0], time_array[-1]), t_eval = time_array, y0 = y0, method = 'RK45', args = arguments)
#Get results
y = [sol.y[:N, :], sol.y[N:2*N, :], sol.y[2*N:3*N, :], sol.y[3*N:4*N, :], sol.y[4*N:5*N, :], sol.y[5*N:6*N, :], sol.y[6*N:7*N, :], sol.y[7*N:8*N, :], sol.y[8*N:9*N, :], sol.y[9*N:, :].reshape(N, N, sol.t.shape[0])]


def plot_spike_train(M):
  y_array = []
  x_array = []
  for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if(M[i,j] == 1):
            y_array.append(i)
            x_array.append(j)

  plt.scatter(x_array, y_array, color ='k',  s = 10)
  plt.ylabel('Neuron #')
  plt.xlabel('Samples')
  plt.xlim(0, M.shape[1])


##Plotting of neuronal spiking.
neuron_array = range(0, N)

spike_matrix = np.array([spike_boolean(y[0][i, :]) for i in neuron_array])
plt.figure()
plot_spike_train(spike_matrix)


##Subplots widget.
n = 5
plt.figure()
plt.subplot(5,1,1)
plt.plot(y[0][n, :])

plt.subplot(5,1,2)
plt.scatter(range(0, len(time_array)),spike_matrix[n])

plt.subplot(5,1,3)
plt.plot(y[5][n, :])

plt.subplot(5,1,4)
plt.plot(y[6][n, :])
plt.plot(y[7][n, :])

plt.subplot(5,1,5)
plt.plot(y[4][n, :])


n = 24
plt.figure()
plt.subplot(4,1,1)
plt.plot(y[0][n, :])
plt.subplot(4,1,2)
plt.scatter(range(0, len(time_array)),spike_matrix[n])
plt.subplot(4,1,3)
plt.plot(y[5][n, :])
plt.subplot(4,1,4)
plt.plot(y[6][n, :])
plt.plot(y[7][n, :])
plt.subplot(6,1,4)
plt.plot(y[4][n, :])