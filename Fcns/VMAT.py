# Vesicular monoamine transporter, 5ht -> v5ht.
# b = cDA
#c= vda
#UNITS IN uM and uM/h. 
def VMAT(b,c):
  k = 0.2 
  V = 1230 
  a = (V*b/(k + b))-(1)*c 
  
  if a < 0:
    a = 0 #Make sure a is not negative. 

  return a
