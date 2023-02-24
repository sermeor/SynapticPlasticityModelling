## Function of  the putative HA transporter in glia. 
# UNITS in uM/h. 
# b = eha
def VHATg(b):
  k = 10
  V = 13500
  a = (V*b/(k + b))
  return a