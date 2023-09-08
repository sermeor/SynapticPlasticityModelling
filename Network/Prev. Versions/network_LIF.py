import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import math

## Mathematical model of synaptogenesis. 
  

def spike_boolean(Vm, Vth):
  return np.array(Vm >= Vth, dtype = int)


## ODE model
def comp_model(t, y, N, NE, th, Vth, Vrest, Vreset):
  #Structure flatten y into list of arrays. 
  y = [y[:N], y[N:2*N], y[2*N:3*N], y[3*N:4*N], y[4*N:].reshape(N, N)]

  # Parameters of LIF model
  tau_m = 10 #ms
  Rm = 10   # membrane resistance (Mohm)
  C_m = 1.0  # membrane capacitance (uF/cm^2)
  

  #Constants
  a1 = 10 #Weight of excitatory AMPA inputs.  
  a2 = 5 #Weight of inhibitory GABA A inputs. 
  a3 = 10 #Weight of excitatory NMDA inputs. 
  a4 = 5 #Weight of inhibitory GABA B inputs. 


  a5 = 0.1 #Rate of production of neurotransmitter per spike (ms-1). 
  a6 = 0.1 #Rate of reuptake of neurotransmitter (ms-1). 
  a7 = 1 #Rate of increase of excitatory w_fast (AMPA) from neurotransmitter presence (ms-1). 
  a8 = 1 #Rate of decrease of excitatory w_fast (AMPA) (ms-1).
  a9 = 1 #Rate of increase of inhibitory w_fast (GABA A) from neurotransmitter presence (ms-1). 
  a10 = 1 #Rate of decrease of inhibitory w_fast (GABA A) (ms-1).
  a11 = 0.1 #Rate of increase of excitatory w_slow (NMDA) from neurotransmitter presence (ms-1). 
  a12 = 0.1 #Rate of decrease of excitatory w_slow (NMDA) (ms-1).
  a13 = 0.05 #Rate of increase of inhibitory w_slow (GABA B) from neurotransmitter presence (ms-1). 
  a14 = 0.05 #Rate of decrease of inhibitory w_slow(GABA B) from neurotransmitter presence (ms-1). 

  #Variables. 
  spike = spike_boolean(y[0], Vth) #Discrete spikes.

  g_AMPA = np.array([a1 * sum(y[4][i,:NE] * y[2][:NE]) for i in range(0, N)])/NE # Conductance factor of AMPA channels. 
  g_GABA_A = np.array([a2 * sum(y[4][i,NE:] * y[2][NE:]) for i in range(0, N)])/(N-NE) # Conductance factor of GABA A channels.
  g_NMDA = np.array([a3 * sum(y[4][i,:NE] * y[3][:NE]) for i in range(0, N)])/NE # Conductance factor of NMDA channels. 
  g_GABA_B = np.array([a4 * sum(y[4][i,NE:] * y[3][NE:]) for i in range(0, N)])/(N-NE) # Conductance factor of GABA B channels.

  k1 = 1 #Coefficient of AMPA effect on membrane potential.
  k2 = 1 #Coefficient of NMDA effect on membrane potential. 
  k3 = 1 #Coefficient of GABA A effect on membrane potential.
  k4 = 1 #Coefficient of GABA B effect on membrane potential.


  #Initialize differential list. 
  dy = [0, 0, 0, 0, 0]

  #Variable in ODE. 
  #y[0] = Vm, membrane potential of neurons. 
  #y[1] = N, Neurotransmitter activity in synapse. 
  #y[2] = w_fast, synaptic weights of activation of fast receptors (AMPA, GABA A). 
  #y[3] = w_slow, synaptic weights of activation of slow receptors (NMDA, GABA B). 
  #y[4] = connectivity matrix.


  # Define Poisson input parameters
  rate = 10.0  # firing rate (Hz)
  w1 = 1.0  # Excitatory synaptic weight
  w2 = 0.1 #Inhibitory synaptic weight
  I_noise_e = w1 * np.random.poisson(rate*th, N)  #Poisson excitatory input noise.
  I_noise_i = w2 * np.random.poisson(rate*th, N)  #Poisson inhibitory input noise.


  dy[0] = (-(y[0] - Vrest) + (Rm/C_m)*(I_noise_e - I_noise_i  + k1*g_AMPA + k2*g_NMDA - k3*g_GABA_A - k4*g_GABA_B))/tau_m 
  dy[1] = a5 * spike - a6 * y[1]
  dy[2] = np.array([a7 * y[1][i] - a8 * y[2][i] if i<NE else a9 * y[1][i] - a10 * y[2][i] for i in range(0, N)])
  dy[3] =  np.array([a11 * y[1][i] - a12 * y[3][i] if i<NE else a13 * y[1][i] - a14 * y[3][i] for i in range(0, N)])
  dy[4] = np.zeros([N, N]).flatten()


  #flatten dy
  dy = np.concatenate(dy).flatten()
  return dy



#Constant parameters
N = 100 #Number of neurons. 
NE = int(0.8*N) #Number of excitatory neurons.


random_seed = 25 #Seed of the pseudo-random number generator. 
np.random.seed(random_seed)


#Time array.
t_factor = 1 # Time factor for graphs.
time = 1000/t_factor # Time of simulation depending on t_factor.
sampling_rate = 10*t_factor #number of samples per time factor units.
time_array = np.linspace(0, time, math.floor(time * sampling_rate + 1))
th = time_array[1] - time_array[0] #Time interval between samples.

#Initial conditions with their original shapes. 
#Vm = np.random.uniform(low=-80, high=-50, size=(N,)) #Membrane potential. 
Vm = np.random.uniform(low=-80, high=-40, size=(N,))
N_0 = np.zeros(N)  #Neurotransmitter. 
g_fast_0 = np.zeros(N) #Fast conductances. 
g_slow_0 = np.zeros(N) #Membrane potential. #Slow conductances. 
C = 1*np.random.rand(N, N).flatten() #Connectivity matrix.

y0 = np.concatenate((Vm, N_0, g_fast_0, g_slow_0, C), axis = 0) #Flatten initial conditions.


#Get solution of the differential equation.
Vth = -50   # threshold potential (mV)
Vrest = -60  # Resting potential (mV)
Vreset = -60  # Reset potential (mV)

y_mat = np.zeros((len(y0), len(time_array)))
y_mat[:, 0] = y0

for i in range(0, len(time_array)-1):
  dy = comp_model(time_array[i], y_mat[:, i], N, NE, th, Vth, Vrest, Vreset)
  y_mat[:, i+1] = y_mat[:, i] + dy*th
  for j in range(0, N):
    if y_mat[j, i] >= Vth:
      y_mat[j, i+1] = Vreset


#Get results.
sol_y = [y_mat[:N, :], y_mat[N:2*N, :],y_mat[2*N:3*N, :], y_mat[3*N:4*N, :]]





def plot_spike_train(M):
  for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if(M[i,j] == 1):
            x1 = [i,i+0.5]
            x2 = [j,j]
            plt.plot(x2,x1,color = 'black')
            plt.ylabel('Neuron #')
            plt.xlabel('Samples')
            plt.xlim(0, M.shape[1])


##Plotting of neuronal spiking. 
neuron_array = range(0, N)
spike_matrix = np.array([spike_boolean(sol_y[0][i, :], Vth) for i in neuron_array])

plt.figure()
plot_spike_train(spike_matrix)

##Subplotting
n = 85
plt.figure()
plt.subplot(4,1,1)
plt.plot(sol_y[0][n, :])
plt.subplot(4,1,2)
plt.scatter(range(0, len(time_array)), spike_matrix[n])

plt.subplot(4,1,3)
plt.plot(sol_y[1][n, :])

plt.subplot(4,1,4)
plt.plot(sol_y[2][n, :])
plt.plot(sol_y[3][n, :])


