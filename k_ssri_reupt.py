## Function of SSRI decrease of SERT ratio. 
# units in uM. 
def k_ssri_reupt(ssri):
  max_r = 2 # Max increase of speed.
  b = 2.5 # Strength.
  ratio =  b * ssri
  if ratio > max_r:
    ratio = max_r
  return ratio
