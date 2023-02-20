## Serotonin synthesis inhibition from 5-HT1B autoreceptors.
# depends on levels of serotonin-activated g-protein (b)
# and the  equilibrium  value  of  the  activated  G-protein (c).
# UNITS IN ratio. 
def inhibsyn5HTto5HT(b, c):
  min_a = 0
  a = 1 - (0.1)*(b - c)
  if a < min_a:
    a = min_a
  return a
