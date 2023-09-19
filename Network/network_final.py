import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math
from numba import njit, jit

## Mathematical model of synaptogenesis.
##Function of rate constant increase of m.
@njit
def alpha_m(Vm):
  return 0.1 * (Vm + 40.0) / (1.0 - np.exp(-(Vm + 40.0) / 10.0))


##Function of rate constant decrease of m.
@njit
def beta_m(Vm):
  return 4.0 * np.exp(-(Vm + 65.0) / 18.0)


##Function of rate constant increase of h.
@njit
def alpha_h(Vm):
  return 0.07 * np.exp(-(Vm + 65.0) / 20.0)


##Function of rate constant decrease of h.
@njit
def beta_h(Vm):
  return 1.0 / (1.0 + np.exp(-(Vm + 35.0) / 10.0))


##Function of rate constant increase of n.
@njit
def alpha_n(Vm):
  return 0.01 * (Vm + 55.0) / (1.0 - np.exp(-(Vm + 55.0) / 10.0))


##Function of rate constant decrease of n.
@njit
def beta_n(Vm):
  return 0.125 * np.exp(-(Vm + 65) / 80.0)


@njit
def alpha_c(Vm):
  return 0.01 * (Vm + 20.0) / (1.0 - np.exp(-(Vm + 20.0) / 2.5))


@njit
def beta_c(Vm):
  return 0.125 * np.exp(-(Vm + 50.0) / 80.0)


## Function of sodium channel current (mA/cm^2).
@njit
def I_Na(m, h, Vm):
  g_Na = 120.0  # maximum sodium conductance (mS/cm^2)
  E_Na = 50.0  # sodium reversal potential (mV)
  return g_Na * (m**3) * h * (Vm - E_Na)


##Function of potassium channels current (mA/cm^2).
@njit
def I_K(n, Vm):
  E_K = -77.0  # potassium reversal potential (mV)
  g_K = 36.0  # maximum potassium conductance (mS/cm^2)
  return g_K * (n**4) * (Vm - E_K)


##Function of VGCC current.
@njit
def I_VGCC(c, Vm):
  E_Ca = 60  #Reversal potential (mV)
  g_Ca = 0.0000075  # Maximum calcium conductance (mS/cm^2)
  return g_Ca * c * (Vm - E_Ca)


##Function of leakage current (mA/cm^2).
@njit
def I_L(Vm):
  g_L = 0.3  # maximum leak conductance (mS/cm^2)
  E_L = -54.4  # leak reversal potential (mV)
  return g_L * (Vm - E_L)


##Function of AMPA current (mA/cm^2).
@njit
def I_AMPA(g_AMPA, Vm):
  E_AMPA = 0.0  #Reversal potential for AMPA channels (mV)
  return g_AMPA * (Vm - E_AMPA)


##Function of magnesium block of NMDA dependent on voltage (mV).
@njit
def MgB(Vm):
  Mg0 = 2  #mM
  return 1 / (1 + (Mg0 * np.exp(-0.062 * Vm)) / 3.57)


##Function of NMDA channel sodium current density (mA/cm^2).
@njit
def I_NMDA_Na(g_NMDA, Vm):
  P_Na = 1  #Permeability ratio to sodium.
  c = 0.1  #Conversor A/m^2 -> mA/cm^2
  P_NMDA = 10 * 10**(-9)  #m/s
  F = 96485  #C/mol
  R = 8.314  #J/K*mol
  T = 308.15  #K
  Nai = 18  #mM
  Nao = 140  #mM
  V_lim = 100  #mV
  a1 = g_NMDA * c * P_NMDA * P_Na * MgB(Vm) * ((Vm / 1000 * F**2) / (R * T))
  a2 = (Nai - Nao * np.exp(-((Vm / 1000 * F) / (R * T)))) / (1 - np.exp(-((Vm / 1000 * F) / (R * T))))

  a2[Vm > V_lim] = Nai
  a2[Vm < -V_lim] = Nao

  I = a1 * a2

  return I


##Function of NMDA channel potassium current density (mA/(cm^2 m-1))
@njit
def I_NMDA_K(g_NMDA, Vm):
  P_K = 1  #Permeability ratio to potassium.
  c = 0.1  #Conversor A/m^2 -> mA/cm^2
  P_NMDA = 10 * 10**(-9)  #m/s
  F = 96485  #C/mol
  R = 8.314  #J/K*mol
  T = 308.15  #K
  Ki = 140  #mM
  Ko = 5  #mM
  V_lim = 100  #mV
  a1 = g_NMDA * c * P_NMDA * P_K * MgB(Vm) * ((Vm / 1000 * F**2) / (R * T))
  a2 = (Ki - Ko * np.exp(-((Vm / 1000 * F) / (R * T)))) / (1 - np.exp(-((Vm / 1000 * F) / (R * T))))
  a2[Vm > V_lim] = Ki
  a2[Vm < -V_lim] = Ko

  I = a1 * a2

  return I


##Function of NMDA channel calcium current density (mA/cm^2).
@njit
def I_NMDA_Ca(g_NMDA, Vm):
  P_Ca = 10.6  #Permeability ratio to calcium.
  c = 0.1  #Conversor A/m^2 -> mA/cm^2
  P_NMDA = 10 * 10**(-9)  #m/s
  F = 96485  #C/mol
  R = 8.314  #J/K*mol
  T = 308.15  #K
  Cai = 0.0001  #mM
  Cao = 2  #mM
  V_lim = 100  #mV
  a1 = g_NMDA * c * P_NMDA * P_Ca * MgB(Vm) * ((4 * Vm / 1000 * F**2) /(R * T))
  a2 = (Cai - Cao * np.exp(-((2 * Vm / 1000 * F) / (R * T)))) / (1 - np.exp(-((2 * Vm / 1000 * F) / (R * T))))

  a2[Vm > V_lim] = Cai
  a2[Vm < -V_lim] = Cao

  I = a1 * a2

  return I


##Function of total NMDA channel current density (mA/cm^2).
@njit
def I_NMDA(g_NMDA, Vm):
  return I_NMDA_Na(g_NMDA, Vm) + I_NMDA_K(g_NMDA, Vm) + I_NMDA_Ca(g_NMDA, Vm)


##Function of inward calcium rate (uM/ms).
@njit
def inward_Ca(g_NMDA, Vm, c):
  F = 96485  # Faraday Constant (mA*ms/umol).
  d = 8.4e-6  #Distance of membrane shell where calcium ions enter (cm).
  s = 1000  #conversor umol/(cm^3 * ms) to uM/ms.
  return -s * (I_NMDA_Ca(g_NMDA, Vm) + I_VGCC(c, Vm)) / (2 * F * d)


##Function of outward calcium rate (uM/ms).
@njit
def outward_Ca(Cai, Cai_eq):
  c = 0.05  #Rate of calcium pump buffering (ms^-1).
  return + c * (Cai - Cai_eq)


##Function of GABA A current (mA/cm^2).
@njit
def I_GABA_A(g_GABA_A, Vm):
  E_GABA_A = -70.0  #Reversal potential for GABA A channels (mV).
  return g_GABA_A * (Vm - E_GABA_A)


##Function of GABA B current (mA/cm^2).
@njit
def I_GABA_B(g_GABA_B, Vm):
  E_GABA_B = -95.0  #Reversal potential for GABA B channels (mV).
  return g_GABA_B * (Vm - E_GABA_B) / (1.0 + np.exp(-(Vm + 80.0) / 25.0))


##Function that draws a Bernoulli sample from the probability of neuron firing.
#Output is 1 (active) or 0 (inactive). Seed is fixed to have the same results.
@njit
def spike_boolean(Vm):
  Vth = 0
  result = np.empty_like(Vm, dtype=np.int32)
  for i in range(len(Vm)):
    result[i] = 1 if Vm[i] >= Vth else 0
  return result


#Function that updates the connectivity matrix.
@njit
def connectivity_update(C, act, N, NE, w_plas):
  # Postsynaptic update.
  sigma = 0.5 #Equilibrium activity of the neuron.
  w_ex = - 0.00002 #Weight of rate of change of post-synaptic connectivity of the neuron on excitatory plates (- when act>sigma, + when act<sigma, compensation).
  w_inh = 0.00001 #Weight of rate of change of post-synaptic connectivity of the neuron on inhibitory plates (+ when act>sigma, - when act<sigma, compensation).
  post_delta_C_row = (act - sigma)
  ##TESTING
  w_plas = 0
  ##TESTING
  post_delta_C = np.empty((N, N), dtype=np.float32)
  post_delta_C[:NE] = post_delta_C_row * (w_ex + w_plas)
  post_delta_C[NE:] = post_delta_C_row * (w_inh + w_plas)

  # Presynaptic update.
  w = 0.00001 #Weight of rate of change of presynaptic connectivity of the neuron (+ when act>sigma, - when act<sigma).
  pre_delta_C = np.empty((N, N), dtype=np.float32)
  for i in range(N):  #rows
    pre_delta_C[i, :] = w * (act[i] - sigma)

  # Sum both effects.
  delta_C = post_delta_C + pre_delta_C

  # Check it is in [0, 1]
  temp = C + delta_C
  invalid_indices = np.logical_or(temp < 0, temp > 1)
  delta_C = np.where(invalid_indices, 0, delta_C)

  return delta_C



#Function that calculates g_AMPA, from AMPA input weights and connectivity weights.
@njit
def g_AMPA_calc(a1, C, w, N, NE):
  return a1 * np.dot(C[:NE, :].T, w[:NE]) / NE


#Function that calculates g_GABA_A, from GABA A input weights and connectivity weights.
@njit
def g_GABA_A_calc(a2, C, w, N, NE):
  return a2 * np.dot(C[NE:, :].T, w[NE:]) / (N - NE)


#Function that calculates g_NMDA, from NMDA input weights and connectivity weights.
@njit
def g_NMDA_calc(a3, C, w, N, NE):
  return a3 * np.dot(C[:NE, :].T, w[:NE]) / NE


#Function that calculates g_GABA B, from GABA B input weights and connectivity weights.
@njit
def g_GABA_B_calc(a4, C, w, N, NE):
  return a4 * np.dot(C[NE:, :].T, w[NE:]) / (N - NE)


##Function that calculates the w_fast uptake.
@njit
def w_fast_update(wfe, wfe_decay, wfi, wfi_decay, Neurot, w_fast, N, NE):
  alpha_w = np.empty(N, dtype=np.float32)
  alpha_w[:NE] = wfe * Neurot[:NE] - wfe_decay * w_fast[:NE]
  alpha_w[NE:] = wfi * Neurot[NE:] - wfi_decay * w_fast[NE:]
  return alpha_w


##Function that calculates the w_fast uptake.
@njit
def w_slow_update(wse, wse_decay, wsi, wsi_decay, Neurot, w_slow, N, NE):
  alpha_w = np.empty(N, dtype=np.float32)
  alpha_w[:NE] = wse * Neurot[:NE] - wse_decay * w_slow[:NE]
  alpha_w[NE:] = wsi * Neurot[NE:] - wsi_decay * w_slow[NE:]
  return alpha_w


#Function of current given by Poison noise inputs (mA).
@njit
def noise(w, rate, N):
  return w * np.random.poisson(rate, N)


#Function that flattens the input array to comp_model.
@njit
def flatten_y(y, N, var_number):
  return y.reshape(var_number + N, N)


#Function to flatten dy arrays.
@njit
def flatten_dy(dy):
  return dy.flatten()


##Function that sets the random seed on the compiled side.
@njit
def set_seed(seed):
  np.random.seed(seed)
  return


##Function of occupacy ratio of NMDA receptors depending on ketamine and norketamine concentration (uM)
#This function assumes that k and nk don't inhibit each other, and that glutamate concentration has no effect on binding rate of k and nk, since their binding stregth is much greater.
@njit
def inhib_NMDA(k, nk):
  #Hill equation (MM non-competitive inhibition)
  #Not affected by glutamate concentration.
  n_k = 1.4  #Hill number ketamine.
  n_nk = 1.2  #Hill number nor-ketamine.
  Ki_k = 2  #Ketamine concentration for half occupacy (uM)
  Ki_nk = 17  #Norketamine concentration for half occupacy (uM)
  if (k>0) and (nk>0):
    f = 1 / (1 + (Ki_k / k)**n_k) + 1 / (1 + (Ki_nk / nk)**n_nk)
  else:
    f = 0
  return f


##Function of fraction CaMKII bound to Ca2+.
#F: fraction of CaMKII subunits bound to Ca+ /CaM.
@njit
def CaMKII(Cai, Cai_eq): 
  K_H1 = 2  # The Ca2 activation Hill constant of CaMKII in uM.
  b = K_H1 - Cai_eq #Value to set the function as 0.5 at Cai_eq.
  return ((Cai + b) / K_H1)**4 / (1 + (((Cai + b) / K_H1)**4))


##Function of fraction of BDNF bound to TrkB receptor (sigmoid).
#bdnf: levels of BDNF protein in the extracellular space.
@njit
def TrkB(bdnf):
  s = 10 #Slope of sigmoid.
  return 1 / (1 + np.exp(-s*(bdnf - 0.5)))

##Function of fraction of p75_NTR proBDNF-bound (sigmoid).
@njit
def p75_NTR(bdnf, pro_bdnf):
  w_pro_bdnf = 0.9 #Factor of pro_bdnf.
  w_bdnf = 1 -  w_pro_bdnf #Factor of bdnf.
  s = 10 #Slope of sigmoid.
  return 1 / (1 + np.exp(-s*(w_bdnf * pro_bdnf + w_pro_bdnf * bdnf - 0.5)))

##Function of BDNF presence dependent on neuronal factors (sigmoid).
@njit
def bdnf_calc(bAP_events, g_AMPA, Cai, Cai_eq, eht, eht_eq):
  delta_Cai = Cai - Cai_eq
  delta_eht = eht - eht_eq
  w_bAP = 0.1 #Weight of bAP on BDNF release.
  w_g_AMPA = 0.1 #Weight of g_AMPA on BDNF release.
  w_Cai = 0.1 #Weight of Cai increase above equilibrium on BDNF release.
  w_eht = 0.1 #Weight of eht increase above equilibrium on BDNF release.
  s = 10 #Slope of the sigmoid.
  bdnf = 1 / (1 + np.exp(-s*(w_bAP*bAP_events + w_g_AMPA*g_AMPA + w_Cai*delta_Cai + w_eht * delta_eht)))
  return bdnf

##Function to determine if action potential is backpropagating.
@njit
def bAP(Vm, act):
  return np.logical_or(Vm > 35, act > 0.7)


# Weights of BDNF/proBDNF and CaMKII plasticity,
#afecting the rate of change of the connectivity matrix.
#Sigmoid that can go to negative values.
@njit
def plasticity_weights_calc(CaMKII_bound, trkB_bound, p75NTR_bound):
  w_CaMKII_bound = 1 #Weight of CaMKII_bound
  w_trkB_bound = 1 #Weight of trkB_bound
  w_p75NTR_bound = 2 #Weight of p75NTR_bound
  s = 10 #Slope of sigmoid.
  w = 0.00001 #Weight to convert to plasticity from sigmoid.
  plasticity_weights = w/(1 + np.exp(-s*(w_CaMKII_bound*CaMKII_bound + w_trkB_bound*trkB_bound - w_p75NTR_bound*p75NTR_bound))) - 0.5
  return plasticity_weights



# ODE model
@njit
def comp_model(t, y, N, NE):
  #Structure flatten y into 2D array.
  y = flatten_y(y, N, 10)

  # Parameters of HH model
  C_m = 1.0  # membrane capacitance (uF/cm^2)
  Cai_eq = 0.1 #Equilibrium internal calcium (uM)

  #Constants
  a1 = (3) * 0.25  #Weight of excitatory AMPA inputs.
  a2 = (3) * 0.25  #Weight of inhibitory GABA A inputs.
  a3 = 50 * (1.5) * 0.25  #Weight of excitatory NMDA inputs; NMDA currents are defined differently so it's higher than others.
  a4 = 5 * (1.5) * 0.25  #Weight of inhibitory GABA B inputs.

  a5 = 1  #Rate of production of neurotransmitter per spike (ms-1).
  a6 = 0.25  #Rate of reuptake of neurotransmitter (ms-1).

  a7 = 0.05  #Rate of increase of excitatory w_fast (AMPA) from neurotransmitter presence (ms-1).
  a8 = 0.1  #Rate of decrease of excitatory w_fast (AMPA) (ms-1).

  a9 = 0.03  #Rate of increase of inhibitory w_fast (GABA A) from neurotransmitter presence (ms-1).
  a10 = 0.1  #Rate of decrease of inhibitory w_fast (GABA A) (ms-1).

  a11 = 0.025  #Rate of increase of excitatory w_slow (NMDA) from neurotransmitter presence (ms-1).
  a12 = 0.020  #Rate of decrease of excitatory w_slow (NMDA) (ms-1).

  a13 = 0.001  #Rate of increase of inhibitory w_slow (GABA B) from neurotransmitter presence (ms-1).
  a14 = 0.001  #Rate of decrease of inhibitory w_slow(GABA B) from neurotransmitter presence (ms-1).

  a15 = 0.01  #Rate of increase of neuron activity from individual spike (ms-1).
  a16 = 0.00001  #Rate of decrease of neuron activity if neuron does not spike (ms-1).


  #Define Poisson input parameters
  rate = 0.1  # firing rate (ms^-1)
  w1 = 10  # Excitatory noise synaptic weight
  w2 = 2  #Inhibitory noise synaptic weight

  I_noise_e = noise(w1, rate, N)  #Poisson excitatory input noise.
  I_noise_i = noise(w2, rate, N)  #Poisson inhibitory input noise.
    
     
  #NMDA inhibition variables.  
  ketamine = 0 #Ketamine concentration (nM).
  norketamine = 0 #Norketamine concentration (nM).
  NMDA_dependency = np.ones(N) #NMDA dependency of neurons.
  NMDA_dependency[:NE] *= 0.1 #Lower NMDA dependency of excitatory neurons.
  inh_NMDA = inhib_NMDA(ketamine, norketamine)*NMDA_dependency #Inhibition score of NMDA.

  #Conductances.
  g_AMPA = g_AMPA_calc(a1, y[10:], y[7], N, NE)  #Conductance factor of AMPA channels.
  g_GABA_A = g_GABA_A_calc(a2, y[10:], y[7], N, NE)  # Conductance factor of GABA A channels.
  g_NMDA = g_NMDA_calc(a3, y[10:], y[8], N,  NE)*inh_NMDA  # Conductance factor of NMDA channels.
  g_GABA_B = g_GABA_B_calc(a4, y[10:], y[8], N, NE)  # Conductance factor of GABA B channels.

  #PFC to DRN variables. 
  #Input from PFC excitatory neurons to DRN neurons. 
  c_pfc_drn = np.mean(y[9][:NE])

  #Synaptic plasticity variables.
  #Variables to be replaced when in full model. 

  eht = 60 # Serotonin concentration (nM).
  eht_eq = 60 #Serotonin concentration in equilibrium (nM)
  spike = spike_boolean(y[0])  #Discrete spikes.
  CaMKII_bound = CaMKII(y[5], Cai_eq)  #Bound Ca2+ to CaMKII protein (0 to 1).
  bAP_events = bAP(y[0], y[9])  #Backpropagating potential events.
  bdnf_c = bdnf_calc(bAP_events, g_AMPA, y[5], Cai_eq, eht, eht_eq) #Factor of BNDF levels presence in synapse.
  trkB_bound = TrkB(bdnf_c)  #Bound BDNF to trkB, sigmoid.
  p75NTR_bound = p75_NTR(bdnf_c, 1 - bdnf_c)  #Bound proBDNF and BDNF to p75NTR, sigmoid.
  #Weights of plasticity processes (close to 1 when neuron is proactively forming connections,
  #0 when the neuron is slower in making connections).
  w_plas = plasticity_weights_calc(CaMKII_bound, trkB_bound, p75NTR_bound)



  #Initialize differential list.
  dy = np.zeros_like(y, dtype=np.float32)

  #Variable in ODE.
  #y[0] = Vm, membrane potential of neurons (mV).
  #y[1] = m, activation gating variable for the voltage-gated sodium (Na+) channels.
  #y[2] = h, activation gating variable for the voltage-gated potassium (K+) channels.
  #y[3] = n, Inactivation gating variable for the Na+ channels.
  #y[4] = c, activation of gatting variable.
  #y[5] = Cai, internal calcium concentration (uM)
  #y[6] = N, Neurotransmitter activity in synapse.
  #y[7] = w_fast, synaptic weights of activation of fast receptors (AMPA, GABA A).
  #y[8] = w_slow, synaptic weights of activation of slow receptors (NMDA, GABA B).
  #y[9] = Activity state of neuron (low/high state).
  #y[10:] = Connectivity matrix.

  #Differential equations.
  dy[0] = (-I_Na(y[1], y[2], y[0]) - I_K(y[3], y[0]) - I_L(y[0]) - I_VGCC(y[4], y[0]) - I_AMPA(g_AMPA, y[0]) - I_NMDA(g_NMDA, y[0]) - I_GABA_A(g_GABA_A, y[0]) - I_GABA_B(g_GABA_B, y[0]) + I_noise_e - I_noise_i) / C_m
  dy[1] = alpha_m(y[0]) * (1.0 - y[1]) - beta_m(y[0]) * y[1]
  dy[2] = alpha_h(y[0]) * (1.0 - y[2]) - beta_h(y[0]) * y[2]
  dy[3] = alpha_n(y[0]) * (1.0 - y[3]) - beta_n(y[0]) * y[3]
  dy[4] = alpha_c(y[0]) * (1 - y[4]) - beta_c(y[0]) * y[4]
  dy[5] = inward_Ca(g_NMDA, y[0], y[4]) - outward_Ca(y[5], Cai_eq)
  dy[6] = a5 * spike - a6 * y[6]
  dy[7] = w_fast_update(a7, a8, a9, a10, y[6], y[7], N, NE)
  dy[8] = w_slow_update(a11, a12, a13, a14, y[6], y[8], N, NE)
  dy[9] = a15 * spike * (1 - y[9]) - a16 * y[9]
  dy[10:] = connectivity_update(y[10:], y[9], N, NE, w_plas)

  #flatten dy
  dy = flatten_dy(dy)
  return dy


#Constant parameters
N = 100  #Number of neurons.
NE = int(0.8 * N)  #Number of excitatory neurons.

random_seed = 25  #Seed of the pseudo-random number generator.
set_seed(random_seed)  #Set seed in compiled code.
np.random.seed(random_seed)  #Set seed in non-compiled code.

#Time array.
t_factor = 1  # Time factor for graphs.
time = 1*60*60*1000 / t_factor  # Time of simulation depending on t_factor.
sampling_rate = 1 * t_factor  #number of samples per time factor units.
time_array = np.linspace(0, time, math.floor(time * sampling_rate + 1), dtype=np.float32)

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

y0 = np.concatenate((Vm, m, h, n, c, Ca_0, N_0, g_fast_0, g_slow_0, act, C), axis=0)  #Flatten initial conditions.
arguments = (N, NE)  #Constant arguments.


#Get solution of the differential equation and store it.
sample_span = 60*1000  #n samples to solve each step
n_iterations = int(np.ceil(len(time_array)/sample_span)-1) #number of iterations to complete time_array.
i_initial = 55

if i_initial != 0:
  data_last =  np.load('data_iteration_'+str(i_initial-1)+'.npy')
  C_last = np.load('C_interation_'+str(i_initial-1)+'.npy')

  y0[:10*N] = data_last[:, -1]
  y0[10*N:] = C_last

##Solve in chunks.
for i in range(i_initial, n_iterations):
  if i != i_initial:
    y0 = sol.y[:, -1].copy()  #Last solution are initial values.
  #Get new solution.
  sol = solve_ivp(comp_model,
                  t_span=(time_array[i * sample_span], time_array[(i + 1) * sample_span]),
                  t_eval=time_array[i * sample_span:(i + 1) * sample_span],
                  y0=y0,
                  method='RK45',
                  args=arguments)
  #Save to txt.
  np.savetxt('data_iteration_'+str(i)+'.csv', sol.y[:10*N], delimiter=',', fmt = "%.2f")
  np.savetxt('C_interation_'+str(i)+'.csv', sol.y[10*N:, 0], delimiter=',', fmt = "%.2f")
  np.save('data_iteration_'+str(i)+'.npy', sol.y[:10*N])
  np.save('C_interation_'+str(i)+'.npy', sol.y[10*N:, 0])


	#Get results
y = [sol.y[:N, :], sol.y[N:2 * N, :], sol.y[2 * N:3 * N, :],
sol.y[3 * N:4 * N, :], sol.y[4 * N:5 * N, :], sol.y[5 * N:6 * N, :],
  sol.y[6 * N:7 * N, :], sol.y[7 * N:8 * N, :], sol.y[8 * N:9 * N, :],
  sol.y[9 * N:10 * N, :], sol.y[10 * N:, :].reshape(N, N, sol.t.shape[0])
]


def plot_spike_train(M):
  y_array = []
  x_array = []
  for i in range(M.shape[0]):
    for j in range(M.shape[1]):
      if (M[i, j] == 1):
        y_array.append(i)
        x_array.append(j)

  plt.scatter(x_array, y_array, color='k', s=10)
  plt.ylabel('Neuron #')
  plt.xlabel('Samples')
  plt.xlim(0, M.shape[1])
  plt.ylim(0, M.shape[0])


##Plotting of neuronal spiking.
neuron_array = range(0, N)
spike_matrix = np.array([spike_boolean(y[0][i, :]) for i in neuron_array])
plt.figure()
plot_spike_train(spike_matrix)

##Subplots widget.
n = 5
plt.figure()
plt.subplot(5, 1, 1)
plt.plot(y[0][n, :])

plt.subplot(5, 1, 2)
plt.scatter(range(0, sample_span), spike_matrix[n])

plt.subplot(5, 1, 3)
plt.plot(y[6][n, :])

plt.subplot(5, 1, 4)
plt.plot(y[7][n, :])
plt.plot(y[8][n, :])

plt.subplot(5, 1, 5)
plt.plot(y[5][n, :])
plt.show()