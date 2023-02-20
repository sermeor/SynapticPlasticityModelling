## Rate of reuptake from serotonin transporters in serotonin terminals.
# b = e5ht
#UNITS IN uM and uM/h. 
def VSERT(b, sert_density, ssri, ki):
  k = 0.060  
  vmax = 250
  vmax_app = vmax * sert_density
  k_app = k * (1 + ssri/ki)
  a = vmax_app*b/(k_app + b) 
  return a
