## Function of serotonin activation of histamine release.
# Dependent on Serotonin-activated g-protein (b) and the  equilibrium  value
# of  the  activated  G-protein (c).
# Units in uM. 
def activR5HTtoHA(b, c):
  min_a = 0
  a = 1 + (3)*(b - c)
  if a<min_a:
    a = min_a
  return a