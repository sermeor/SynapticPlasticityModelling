# ODE model
@njit
def comp_model(t, y, N, NE):
  #Structure flatten y into 2D array.
  ynn = flatten_y(y[57:], N, 10)

  # Parameters of HH model
  C_m = 1.0  # membrane capacitance (uF/cm^2)
  Cai_eq = 0.1 #Equilibrium internal calcium (uM)

  #Constants
  e1 = (3) * 0.25  #Weight of excitatory AMPA inputs.
  e2 = (3) * 0.25  #Weight of inhibitory GABA A inputs.
  e3 = 50 * (1.5) * 0.25  #Weight of excitatory NMDA inputs; NMDA currents are defined differently so it's higher than others.
  e4 = 5 * (1.5) * 0.25  #Weight of inhibitory GABA B inputs.

  e5 = 1  #Rate of production of neurotransmitter per spike (ms-1).
  e6 = 0.25  #Rate of reuptake of neurotransmitter (ms-1).

  e7 = 0.05  #Rate of increase of excitatory w_fast (AMPA) from neurotransmitter presence (ms-1).
  e8 = 0.1  #Rate of decrease of excitatory w_fast (AMPA) (ms-1).

  e9 = 0.03  #Rate of increase of inhibitory w_fast (GABA A) from neurotransmitter presence (ms-1).
  e10 = 0.1  #Rate of decrease of inhibitory w_fast (GABA A) (ms-1).

  e11 = 0.025  #Rate of increase of excitatory w_slow (NMDA) from neurotransmitter presence (ms-1).
  e12 = 0.020  #Rate of decrease of excitatory w_slow (NMDA) (ms-1).

  e13 = 0.001  #Rate of increase of inhibitory w_slow (GABA B) from neurotransmitter presence (ms-1).
  e14 = 0.001  #Rate of decrease of inhibitory w_slow(GABA B) from neurotransmitter presence (ms-1).

  e15 = 0.01  #Rate of increase of neuron activity from individual spike (ms-1).
  e16 = 0.00001  #Rate of decrease of neuron activity if neuron does not spike (ms-1).


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
  g_AMPA = g_AMPA_calc(e1, ynn[10:], ynn[7], N, NE)  #Conductance factor of AMPA channels.
  g_GABA_A = g_GABA_A_calc(e2, ynn[10:], ynn[7], N, NE)  # Conductance factor of GABA A channels.
  g_NMDA = g_NMDA_calc(e3, ynn[10:], ynn[8], N,  NE)*inh_NMDA  # Conductance factor of NMDA channels.
  g_GABA_B = g_GABA_B_calc(e4, ynn[10:], ynn[8], N, NE)  # Conductance factor of GABA B channels.

  #PFC to DRN variables. 
  #Input from PFC excitatory neurons to DRN neurons. 
  c_pfc_drn = np.mean(ynn[9][:NE])

  #Synaptic plasticity variables.
  #Variables to be replaced when in full model. 

  eht = 60 # Serotonin concentration (nM).
  eht_eq = 60 #Serotonin concentration in equilibrium (nM)
  spike = spike_boolean(ynn[0])  #Discrete spikes.
  CaMKII_bound = CaMKII(ynn[5], Cai_eq)  #Bound Ca2+ to CaMKII protein (0 to 1).
  bAP_events = bAP(ynn[0], ynn[9])  #Backpropagating potential events.
  bdnf_c = bdnf_calc(bAP_events, g_AMPA, ynn[5], Cai_eq, eht, eht_eq) #Factor of BNDF levels presence in synapse.
  trkB_bound = TrkB(bdnf_c)  #Bound BDNF to trkB, sigmoid.
  p75NTR_bound = p75_NTR(bdnf_c, 1 - bdnf_c)  #Bound proBDNF and BDNF to p75NTR, sigmoid.
  #Weights of plasticity processes (close to 1 when neuron is proactively forming connections,
  #0 when the neuron is slower in making connections).
  w_plas = plasticity_weights_calc(CaMKII_bound, trkB_bound, p75NTR_bound)

  #Initialize differential list.
  dynn = np.zeros_like(ynn, dtype=np.float32)

  #Variable in ODE.
  #ynn[0] = Vm, membrane potential of neurons (mV).
  #ynn[1] = m, activation gating variable for the voltage-gated sodium (Na+) channels.
  #ynn[2] = h, activation gating variable for the voltage-gated potassium (K+) channels.
  #ynn[3] = n, Inactivation gating variable for the Na+ channels.
  #ynn[4] = c, activation of gatting variable.
  #ynn[5] = Cai, internal calcium concentration (uM)
  #ynn[6] = N, Neurotransmitter activity in synapse.
  #ynn[7] = w_fast, synaptic weights of activation of fast receptors (AMPA, GABA A).
  #ynn[8] = w_slow, synaptic weights of activation of slow receptors (NMDA, GABA B).
  #ynn[9] = Activity state of neuron (low/high state).
  #ynn[10:] = Connectivity matrix.

  #Differential equations.
  dynn[0] = (-I_Na(ynn[1], ynn[2], ynn[0]) - I_K(ynn[3], ynn[0]) - I_L(ynn[0]) - I_VGCC(ynn[4], ynn[0]) - I_AMPA(g_AMPA, ynn[0]) - I_NMDA(g_NMDA, ynn[0]) - I_GABA_A(g_GABA_A, ynn[0]) - I_GABA_B(g_GABA_B, ynn[0]) + I_noise_e - I_noise_i) / C_m
  dynn[1] = alpha_m(ynn[0]) * (1.0 - ynn[1]) - beta_m(ynn[0]) * ynn[1]
  dynn[2] = alpha_h(ynn[0]) * (1.0 - ynn[2]) - beta_h(ynn[0]) * ynn[2]
  dynn[3] = alpha_n(ynn[0]) * (1.0 - ynn[3]) - beta_n(ynn[0]) * ynn[3]
  dynn[4] = alpha_c(ynn[0]) * (1 - ynn[4]) - beta_c(ynn[0]) * ynn[4]
  dynn[5] = inward_Ca(g_NMDA, ynn[0], ynn[4]) - outward_Ca(ynn[5], Cai_eq)
  dynn[6] = e5 * spike - e6 * ynn[6]
  dynn[7] = w_fast_update(e7, e8, e9, e10, ynn[6], ynn[7], N, NE)
  dynn[8] = w_slow_update(e11, e12, e13, e14, ynn[6], ynn[8], N, NE)
  dynn[9] = e15 * spike * (1 - ynn[9]) - e16 * ynn[9]
  dynn[10:] = connectivity_update(ynn[10:], ynn[9], N, NE, w_plas)

  #flatten dy
  dynn = flatten_dy(dynn)
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

