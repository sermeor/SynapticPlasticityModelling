## Function of histidine transporter. 
# Histidine transport from blood to glia.
# UNITS in uM/h. 
# b = bht.
def VHTLg(b):
  km = 1000
  vmax = 2340
  a = vmax*(b/(km + b))
  return a