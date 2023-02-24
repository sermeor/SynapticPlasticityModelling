## Function of histamine inhibition of histamine firing.
# Dependent on Serotonin-activated g-protein (b) and the  equilibrium  value
# of  the  activated  G-protein (c).
# Units in uM. 
def inhibRHAtoHA(b, c):
  #b = gstar
  min_a = 0
  a = 1 - (2)*(b - c)
  if a<min_a:
    a = min_a
  return a