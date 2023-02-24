## Function of histamine methyltransferase, 
# Histamine metabolisis in glia.
# UNITS in uM/h. 
# b = gha.
def VHNMTg(b):
  k = 4.2
  V = 212  
  a = (V*b/(k + b))
  return a