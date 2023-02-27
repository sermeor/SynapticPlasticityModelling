import numpy as np
from Fcns.activR5HTtoHA import *
from Fcns.allo_ssri_ki import *
from Fcns.degran_ha_mc import *
from Fcns.fireha import *
from Fcns.fireht import *
from Fcns.FMH_inj import *
from Fcns.H1ha import *
from Fcns.H1ht import *
from Fcns.HTDCin import *
from Fcns.inhibR5HTto5HT import *
from Fcns.inhibRHAto5HT import *
from Fcns.inhibRHAtoHA import *
from Fcns.inhibsyn5HTto5HT import *
from Fcns.inhibsynHAtoHA import *
from Fcns.k_5ht1ab_rel_ps import *
from Fcns.k_5ht1ab_rel_sp import *
from Fcns.k_fmh_inh import *
from Fcns.k_ssri_reupt import *
from Fcns.mc_activation import *
from Fcns.SSRI_inj import *
from Fcns.TCcatab import *
from Fcns.VAADC import *
from Fcns.VDRR import *
from Fcns.vha_trafficking import *
from Fcns.VHAT import *
from Fcns.VHATg import *
from Fcns.VHATmc import *
from Fcns.VHNMT import *
from Fcns.VHNMTg import *
from Fcns.VHNMTmc import *
from Fcns.vht_trafficking import *
from Fcns.VHTDC import *
from Fcns.VHTDCg import *
from Fcns.VHTDCmc import *
from Fcns.VHTL import *
from Fcns.VHTLg import *
from Fcns.VHTLmc import *
from Fcns.VMAT import *
from Fcns.VMATH import *
from Fcns.VMATHmc import *
from Fcns.VPOOL import *
from Fcns.VSERT import *
from Fcns.VTPH import *
from Fcns.VTRPin import *
from Fcns.VUP2 import *






def comp_model(y, t, v2, ssri_molecular_weight, SSRI_start_time, SSRI_repeat_time, SSRI_q_inj, 
  fmh_molecular_weight, FMH_start_time, FMH_repeat_time, FMH_q_inj, mc_switch, 
  mc_start_time, btrp0, eht_basal, gstar_5ht_basal, gstar_ha_basal, bht0, 
  vht_basal, vha_basal):

  dy = np.zeros(54)
  
  # Serotonin Terminal Model
  NADP = 26  # NADP concentration in uM.
  NADPH = 330    # NADPH concentration in uM.
  TRPin = 157.6 # addition of tryptophan into blood (uM/h).
  a1 = 5 # Strength of btrp stabilization.  
  a2 = 20    # bound 5ht to autoreceptors produce G5ht*                
  a3 = 200  # T5ht* reverses G5ht* to G5ht.                  
  a4 = 30   # G5ht* produces T5ht*                     
  a5 = 200  # decay of T5ht*.                
  a6 = 36 # rate eht bounding to autoreceptors.                       
  a7 = 20 # rate of 5ht unbonding from autoreceptors.   
  a8 = 1 # 5-HT leakage diffusion from cytosol to extracellular space.
  a9 = 1  # 5-HT leakage diffusion from glia to extracellular space.
  a10 = 1  # catabolism of hiaa
  a11 = 0.01   # UP2 multiplier
  a12 = 2  # removal of trp
  a13 = 1   # removal of pool
  a14 = 40  # eht removal rate.
  a15 = 1 # rate of vht_reserve moving to vht.
  a16 = 1.89 # Factor of release per firing event. 
  a17 = 100  # bound eha to heteroreceptors produce Gha*.
  a18 = 961.094   # Tha* reverses Gha* to Gha.
  a19 = 20  # Gha* produces THA*.
  a20 = 66.2992  # decay of Tha*.
  a21 = 5  # eha binds to heteroreceptors. 
  a22 = 65.6179   # eha dissociates from heteroreceptors.
  g0 = 10 # total g-protein of serotonin autoreceptors.                           
  t0 = 10 # total T protein of serotonin autoreceptors.                           
  b0 = 10 # total serotonin autoreceptors.
  gh0 = 10 # total g-protein of histamine heteroreceptors. 
  th0 =  10  # total T regulary protein of histamine heteroreceptors. 
  bh0 =  10 # total histamine heteroreceptors
  ssri = (y[24]/v2)*1000/(ssri_molecular_weight) # SSRI concentration from compartmental model in uM -> umol/L. 

  # Parameters for SERT membrane, inactivity and pool transport.
  k_ps = 10 * k_5ht1ab_rel_ps(y[11], gstar_5ht_basal)
  k_sp = 10 * k_5ht1ab_rel_sp(y[11], gstar_5ht_basal)
  k_si = 7.5 * k_ssri_reupt(ssri)
  k_is = 0.75
  
  # Equation parameters.
  # y[0] = btrp
  # y[1] = bh2
  # y[2] = bh4
  # y[3] = trp
  # y[4] = htp
  # y[5] = cht
  # y[6] = vht (needs fitting)
  # y[7] = vht_reserve (needs fitting)
  # y[8] = eht
  # y[9] = hia (needs fitting)
  # y[10] = trppool
  # y[11] = gstar
  # y[12] = tstar
  # y[13] = bound
  # y[14] = glialht (needs fitting)
  # y[15] = Gha*
  # y[16] = Tha*
  # y[17] = bound ha
  # y[18] = SERT_surface_phospho
  # y[19] = SERTs_surface
  # y[20] = SERT_pool
  # y[21] = SERT_inactive
  
  dy[0] = TRPin - VTRPin(y[0]) - a1 * (y[0] - btrp0)
  dy[1] = inhibsyn5HTto5HT(y[11], gstar_5ht_basal) * VTPH(y[3], y[2]) - VDRR(y[1], NADPH, y[2], NADP)
  dy[2] = VDRR(y[1], NADPH, y[2], NADP) - inhibsyn5HTto5HT(y[11], gstar_5ht_basal) * VTPH(y[3], y[2]) 
  dy[3] = VTRPin(y[0]) - inhibsyn5HTto5HT(y[11], gstar_5ht_basal) * VTPH(y[3], y[2]) - VPOOL(y[3], y[10]) - a12 * y[3]
  dy[4] = inhibsyn5HTto5HT(y[11], gstar_5ht_basal) * VTPH(y[3], y[2]) - VAADC(y[4])
  dy[5] = VAADC(y[4]) - VMAT(y[5], y[6]) - VMAT(y[5], y[7]) + VSERT(y[8], y[19], ssri, allo_ssri_ki(ssri)) - TCcatab(y[5]) - a15 * (y[5] - y[8])
  dy[6] = VMAT(y[5], y[6]) - a16 * fireht(t, inhibR5HTto5HT(y[11], gstar_5ht_basal) * inhibRHAto5HT(y[15], gstar_ha_basal)) * y[6] + vht_trafficking(y[6], vht_basal)
    
  dy[7] = VMAT(y[5], y[7]) - a15 * vht_trafficking(y[6], vht_basal)


    
  dy[8] = a16 * fireht(t, inhibR5HTto5HT(y[11], gstar_5ht_basal) * inhibRHAto5HT(y[15], gstar_ha_basal)) * y[6] - VSERT(y[8], y[19], ssri, allo_ssri_ki(ssri)) - a11 * H1ht(y[8], eht_basal) * VUP2(y[8]) - a14 * y[8] + a8 * (y[5] - y[8]) + a9 * (y[14] - y[8]) 



    













    
  dy[9] = TCcatab(y[5]) + TCcatab(y[14]) - a10 * y[9] 
  dy[10] = VPOOL(y[3], y[10]) - a13 * y[10]
  dy[11] = a2 * y[13]**2 * (g0 - y[11]) - a3 * y[12] * y[11]  
  dy[12] = a4 * y[11]**2 * (t0 - y[12]) - a5 * y[12]
  dy[13] = a6 * y[8] * (b0 - y[13]) - a7 * y[13] 
  dy[14] = a11 * H1ht(y[8], eht_basal) * VUP2(y[8]) - TCcatab(y[14]) - a9 * (y[14] - y[8])  
  dy[15] = a17 * y[17]**2 * (gh0 - y[15]) - a18 * y[16] * y[15]
  dy[16] = (a19*y[15]**2 * (th0 - y[16])  - a20 * y[16]) 
  dy[17] = (a21*y[29]*(bh0 - y[17])  - a22*y[17]) 
  dy[18] = k_ps * y[20] - k_sp * y[18]
  dy[19] = dy[18] - k_si * y[19] + k_is * y[21]
  dy[20] = k_sp * y[18]  - k_ps * y[20]
  dy[21] = k_si * y[19]  - k_is * y[21]


    


  # Rates between comparments (h-1).
  k01 = 0.6
  k10 = 3
  k12 = 9.9
  k21 = 2910
  k13 = 6
  k31 = 0.6
  
  # Parameters.
  protein_binding = 0.56
  protein_brain_binding = 0.15
  
  # y[22] = Peritoneum concentration in ug.
  # y[23] = Blood concentration in ug.
  # y[24] = Brain concentration in ug.
  # y[25] = Periphery concentration in ug.
  
  # Differential equations.
  dy[22] = 0#SSRI_inj(t, SSRI_start_time, SSRI_repeat_time, SSRI_q_inj) - k01*(y[22])
  dy[23] = 0#k01*(y[22]) - (k10 + k12)*(y[23]*(1-protein_binding)) + k21*(y[24]*(1-protein_brain_binding)) - k13*(y[23]*(1-protein_binding)) + k31*(y[25])
  dy[24] = 0#k12*(y[24]*(1-protein_binding)) - k21*(y[24]*(1-protein_brain_binding))
  dy[25] = 0#k13*(y[23]*(1-protein_binding)) - k31*(y[25])

  ## Histamine Terminal Model. 
  b1 = 15  #HA leakage from the cytosol to the extracellular space. 
  b2 = 3.5 #HA release per action potential. 
  b3 = 15 #HA leakage from glia to the extracellular space.
  b4 = 0.05 #HA removal from the extracellular space
  b5 = 0.25  #Strength of stabilization of blood HT near 100μM. 
  b6 = 2.5 #From cHT to HTpool.
  b7 = 1 #From HTpool to cHT. 
  b8 = 1 #Other uses of HT remove HT. 
  b9 = 1  #From gHT to gHTpool. 
  b10 = 1 #From gHTpool to gHT.  
  b11 = 1 #Removal of gHT or use somewhere else. 
  b12 = 10 #Factor of activation of glia histamine production. 
  b13 = 100 #Bound eha to autoreceptors produce G∗. 
  b14 = 961.094 #T∗ facilitates the reversion of G∗ to G. 
  b15 = 20 #G∗ produces T∗. 
  b16 = 66.2992 #decay coefficient of T∗
  b17 = 5  #eha binds to autoreceptors. 
  b18 = 65.6179 #eha dissociates from autoreceptors.
  b19 = 20  #bound e5ht to heteroreceptors to produce G5ht* 
  b20 = 200  #T5ht* reverses G5ht* to G5ht.
  b21 = 30   #G5ht* produces T5ht* 
  b22 =  200  #decay of T5ht*.
  b23 =  36  #rate eht bounding to heteroreceptors.
  b24 =  20  #rate of 5ht unbonding from heteroreceptors. 
  g0HH = 10  #Total gstar for H3 on HA neuron
  t0HH = 10 #Total tstar for H3 autoreceptors on HA neuron
  b0HH = 10  #Total H3 autoreceptors on HA neuron
  g05ht = 10 # total g-protein of serotonin heteroreceptors in histamine varicosity.                           
  t05ht = 10 # total T protein serotonin heteroreceptors in histamine varicosity.                           
  b05ht = 10 # total serotonin heteroreceptors in histamine varicosities.
  HTin = 636.5570 # Histidine input to blood histidine uM/h. 
    
  # y[26]= cha
  # y[27] = vha 
  # y[28] = vha_reserve
  # y[29] = eha
  # y[30] = gha 
  # y[31] = bht 
  # y[32] = cht 
  # y[33] = chtpool 
  # y[34] = gstar 
  # y[35] = tstar 
  # y[36] = bound 
  # y[37] = gstar5ht 
  # y[38] = tstar5ht 
  # y[39] = bound5ht 
  # y[40] = ght
  # y[41] = ghtpool


  dy[26] = 0#inhibsynHAtoHA(y[34], gstar_ha_basal) * y[53] * VHTDC(y[32]) - VMATH(y[26], y[27]) - VHNMT(y[26]) - b1 * (y[26] - y[29]) + VHAT(y[29]) - VMATH(y[26], y[28])
  dy[27] = 0#VMATH(y[26], y[27]) - fireha(t, inhibRHAtoHA(y[34], gstar_ha_basal) * activR5HTtoHA(y[37], gstar_5ht_basal)) * b2 * y[27] + vha_trafficking(y[27], vha_basal)
  dy[28] = 0#VMATH(y[26], y[28]) - vha_trafficking(y[27], vha_basal)
  dy[29] = 0#fireha(t, inhibRHAtoHA(y[34], gstar_ha_basal) * activR5HTtoHA(y[37], gstar_5ht_basal)) * b2 * y[27] - VHAT(y[29]) + b3 * (y[30] - y[29]) + b1 * (y[26] - y[29]) - H1ha(y[29]) * VHATg(y[29]) - b4 * y[29] - mc_activation(t, mc_switch, mc_start_time) * VHATmc(y[29]) + inhibRHAtoHA(y[46], gstar_ha_basal) * degran_ha_mc(mc_activation(t, mc_switch, mc_start_time)) * y[45]
  dy[30] = 0#H1ha(y[29]) * VHATg(y[29]) - b3 * (y[30] - y[29]) - VHNMTg(y[30]) + (1 + b12 * mc_activation(t, mc_switch, mc_start_time)) * y[53] * VHTDCg(y[40])
  dy[31] = 0#HTin - VHTL(y[31]) - VHTLg(y[31]) - b5 * (y[31] - bht0) - mc_activation(t, mc_switch, mc_start_time) * VHTLmc(y[31])
  dy[32] = 0#VHTL(y[31]) - inhibsynHAtoHA(y[34], gstar_ha_basal) * y[53] * VHTDC(y[32]) - b6 * y[32] + b7 * y[33]
  dy[33] = 0#b6 * y[32] - b7 * y[33] - b8 * y[33]
  dy[34] = 0#b13 * y[36]**2 * (g0HH - y[34]) - b14 * y[35] * y[34]
  dy[35] = 0#b15 * y[34]**2 * (t0HH - y[35]) - b16 * y[35]
  dy[36] = 0#b17 * y[29] * (b0HH - y[36]) - b18 * y[36]
  dy[37] = 0#b19 * y[39]**2 * (g05ht - y[37]) - b20 * y[38] * y[37]


  # Mast Cell Model
  # y[42] = cht. 
  # y[43] = chtpool.
  # y[44] = cha. 
  # y[45] = vha. 
  # y[46] =  Gha*.
  # y[47] = Tha*.
  # y[48]  =  bound ha.
  
  c1 = 1 #From cHT to HTpool.
  c2 = 1 #From HTpool to cHT. 
  c3 = 1 #Removal of cHT or use somewhere else. 
  c4 = 100 # Bound autoreceptors produce g*. 
  c5 = 961.094 #T∗ facilitates the reversion of G∗ to G.
  c6 = 20 #G∗ produces T∗.
  c7 = 66.2992 #decay coefficient of T∗
  c8 = 5  #eHA binds to autoreceptors. 
  c9 = 65.6179 #eHA dissociates from autoreceptors
  g0Hmc = 10  #Total gstar for H3 on mast cell.
  t0Hmc = 10 #Total tstar for H3 on mast cell.
  b0Hmc = 10  #Total H3 receptors on mast cell.

  dy[42] = 0#mc_activation(t, mc_switch, mc_start_time) * VHTLmc(y[31]) - inhibsynHAtoHA(y[46], gstar_ha_basal) * y[53] * VHTDCmc(y[42]) - c1 * y[42] + c2 * y[43]
  dy[43] = 0#c1 * y[42] - c2 * y[43] - c3 * y[43]
  dy[44] = 0#inhibsynHAtoHA(y[46], gstar_ha_basal) * y[53] * VHTDCmc(y[42]) - VMATHmc(y[44], y[45]) - VHNMTmc(y[44]) + mc_activation(t, mc_switch, mc_start_time) * VHATmc(y[29])
  dy[45] = 0#VMATHmc(y[44], y[45]) - inhibRHAtoHA(y[46], gstar_ha_basal) * degran_ha_mc(mc_activation(t, mc_switch, mc_start_time)) * y[45]
  dy[46] = 0#c4 * y[48]**2 * (g0Hmc - y[46]) - c5 * y[47] * y[46]
  dy[47] = 0#c6 * y[46]**2 * (t0Hmc - y[47]) - c7 * y[47]
  dy[48] = 0#c8 * y[29] * (b0Hmc - y[48]) - c9 * y[48]


  ## FMH Pharmacokinetics Model
  # Rates between comparments (h-1). 
  k01f = 3.75
  k10f = 1.75
  k12f = 0.1875
  k21f = 3.5
  k13f = 5
  k31f = 1.5
  
  #Parameters.
  protein_binding_fmh = 0.60
  protein_brain_binding_fmh = 0.15
  fmh = (y[51]/v2)*1000/(fmh_molecular_weight) # Concentration of FMH in uM -> umol/L. 
  #y[49] = Peritoneum concentration in ug.
  #y[50] = Blood concentration in ug.
  #y[51] = Brain concentration in ug.
  #y[52] = Periphery concentration in ug. 
  #y[53] = Ratio of active HTDC in cytosol of histamine, glia and mast cells.

  dy[49] = 0#FMH_inj(t, FMH_start_time, FMH_repeat_time, FMH_q_inj) - k01f * y[49]
  dy[50] = 0#k01f * y[49] - (k10f + k12f) * (y[50] * (1 - protein_binding_fmh)) + k21f * (y[51] * (1 - protein_brain_binding_fmh)) - k13f * (y[50] * (1 - protein_binding_fmh)) + k31f * (y[52])
  dy[51] = 0#k12f * (y[50] * (1 - protein_binding_fmh)) - k21f * (y[51] * (1 - protein_brain_binding_fmh))
  dy[52] = 0#k13f * (y[50] * (1 - protein_binding_fmh)) - k31f * y[52]
  dy[53] = 0#-k_fmh_inh(fmh) * y[53] + HTDCin(y[53])

  return dy

