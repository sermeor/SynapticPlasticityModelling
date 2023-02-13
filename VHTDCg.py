## Function of histidine decarboxylase in glia. 
# Histamine synthesis in glia.
# UNITS in uM/h. 
# b = gha.
def VHTDCg(b):
  # b = cht
  #  c = G*
  km = 270
  v = 61.4250
  a =  v*(b/(km + b)) 
  return a