## Autoreceptor 5-HT1B control of SERT density, from pool to surface. 
#Likely mediated by protein kinases. 5ht1b activation increases g-coupled
#protein which downregulates PKA, which reduces the phosporilation of SERTs
#(removal) and increases the density of SERTs. 
def k_5ht1ab_rel_ps(g, gbasal):
  gdiff = g - gbasal
  max_r = 2 # Max increase of speed.
  min_r = 0 # Min value of speed.
  b = 7.5 # Strength. 
  ratio = 1 + b * gdiff
  if ratio > max_r:
    ratio = max_r
  elif ratio < min_r:
    ratio = min_r
  
  return ratio