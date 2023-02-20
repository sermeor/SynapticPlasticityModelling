# Rate of production of c5ht from 5htp. 
#Follows conventional michaelis menten mechanism. 
# b = 5htp
#UNITS in uM and uM/h. 
def VAADC(b):
  km  = 160 
  vmax = 400  
  a = vmax*b/(km + b)
  return a