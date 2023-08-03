#Tester
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import math

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

def index(i, N):
  new_i = [x for x in range(i*N, (i+1)*N)]

  return new_i

## ODE model
def comp_model(t, y, N, NE, th):
  #Structure flatten y into list of arrays.
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


  spike = spike_boolean(y[index(0, N)]) #Discrete spikes.

  # Define Poisson input parameters
  rate = 10.0  # firing rate (Hz)
  w1 = 0.75  # Excitatory noise synaptic weight
  w2 = 0.2 #Inhibitory noise synaptic weight
  I_noise_e = w1 * np.random.poisson(rate*th, N)  #Poisson excitatory input noise.
  I_noise_i = w2 * np.random.poisson(rate*th, N)  #Poisson inhibitory input noise.

  g_AMPA = np.array([a1 * sum(y[index(8+i, N)][:NE] * y[index(6, N)][:NE]) for i in range(0, N)])/NE # Conductance factor of AMPA channels.
  g_GABA_A = np.array([a2 * sum(y[index(8+i, N)][NE:] * y[index(6, N)][NE:]) for i in range(0, N)])/(N-NE) # Conductance factor of GABA A channels.
  g_NMDA = np.array([a3 * sum(y[index(8+i, N)][:NE] * y[index(7, N)][:NE]) for i in range(0, N)])/NE # Conductance factor of NMDA channels.
  g_GABA_B = np.array([a4 * sum(y[index(8+i, N)][NE:] * y[index(7, N)][NE:]) for i in range(0, N)])/(N-NE) # Conductance factor of GABA B channels.


  #Initialize differential list.
  #dy = [0, 0, 0, 0, 0, 0, 0, 0, 0]
  dy = np.zeros(((8+N)* 25))


  contvty_update = np.zeros((N*N))

  #Variable in ODE.
  #y[0] = Vm, membrane potential of neurons.
  #y[1] = m, activation gating variable for the voltage-gated sodium (Na+) channels.
  #y[2] = h, activation gating variable for the voltage-gated potassium (K+) channels.
  #y[3] = n, Inactivation gating variable for the Na+ channels.
  #y[4] = Cai, internal calcium concentration (uM)
  #y[5] = N, Neurotransmitter activity in synapse.
  #y[6] = w_fast, synaptic weights of activation of fast receptors (AMPA, GABA A).
  #y[7] = w_slow, synaptic weights of activation of slow receptors (NMDA, GABA B).
  #y[8] = connectivity matrix.

  #Differential equations
  dy[index(0,N)] = (- I_Na(y[index(1, N)], y[index(2, N)], y[index(0, N)]) - I_K(y[index(3, N)], y[index(0, N)]) - I_L(y[index(0, N)]) - I_AMPA(g_AMPA, y[index(0, N)]) - I_NMDA(g_NMDA, y[index(0, N)]) - I_GABA_A(g_GABA_A, y[index(0, N)]) - I_GABA_B(g_GABA_B, y[index(0, N)]) + I_noise_e - I_noise_i)/C_m
  dy[index(1,N)] = alpha_m(y[index(0, N)])*(1.0 - y[index(1, N)]) - beta_m(y[index(0, N)])*y[index(1, N)]
  dy[index(2,N)] = alpha_h(y[index(0, N)])*(1.0 - y[index(2, N)]) - beta_h(y[index(0, N)])*y[index(2, N)]
  dy[index(3,N)] = alpha_n(y[index(0, N)])*(1.0 - y[index(3, N)]) - beta_n(y[index(0, N)])*y[index(3, N)]
  dy[index(4,N)] = inward_Ca(g_NMDA, y[index(0, N)]) - outward_Ca(y[index(4, N)])

  dy[index(5,N)] = a5 * spike - a6 * y[index(5, N)]


  dy[index(6,N)] = np.array([a7 * y[index(5, N)][i] - a8 * y[index(6, N)][i] if i<NE else a9 * y[index(5, N)][i] - a10 * y[index(6, N)][i] for i in range(0, N)])

  dy[index(7,N)] =  np.array([a11 * y[index(5, N)][i] - a12 * y[index(7, N)][i] if i<NE else a13 * y[index(5, N)][i] - a14 * y[index(7, N)][i] for i in range(0, N)])

  dy[8*N:] = contvty_update

  return dy


#Constant parameters
N = 25 #Number of neurons.
NE = int(0.8*N) #Number of excitatory neurons.

random_seed = 25 #Seed of the pseudo-random number generator.
np.random.seed(random_seed)


#Time array.
t_factor = 1 # Time factor for graphs.
time = 2000/t_factor # Time of simulation depending on t_factor.
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
C = 10*np.random.rand(N, N).flatten() #Connectivity matrix.

y0 = np.concatenate((Vm, m, h, n, Ca_0, N_0, g_fast_0, g_slow_0, C), axis = 0) #Flatten initial conditions.

arguments = (N, NE,  th)

#Get solution of the differential equation.
sol = solve_ivp(comp_model, t_span = (time_array[0], time_array[-1]), t_eval = time_array, y0 = y0, method = 'RK45', args = arguments)
#Get results
y = [sol.y[:N, :], sol.y[N:2*N, :], sol.y[2*N:3*N, :], sol.y[3*N:4*N, :], sol.y[4*N:5*N, :], sol.y[5*N:6*N, :], sol.y[6*N:7*N, :], sol.y[7*N:8*N, :], sol.y[8*N:, :].reshape(N, N, sol.t.shape[0])]


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
print('Excitatory')
plt.subplot(5,1,1)
plt.plot(y[0][n, :])
plt.subplot(5,1,2)
plt.scatter(range(0, len(time_array)),spike_matrix[0])
plt.subplot(5,1,3)
plt.plot(y[5][n, :])
plt.subplot(5,1,4)
plt.plot(y[6][n, :])
plt.plot(y[7][n, :])
plt.subplot(5,1,5)
plt.plot(y[4][n, :])


n = 24
plt.figure()
print('Inhibitory')
plt.subplot(4,1,1)
plt.plot(y[0][n, :])
plt.subplot(4,1,2)
plt.scatter(range(0, len(time_array)),spike_matrix[0])
plt.subplot(4,1,3)
plt.plot(y[5][n, :])
plt.subplot(4,1,4)
plt.plot(y[6][n, :])
plt.plot(y[7][n, :])
plt.subplot(6,1,4)
plt.plot(y[4][n, :])