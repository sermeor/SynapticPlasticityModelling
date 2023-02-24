# tryptophan transporter rate from blood to cytosol. 
# b = btyr
# UNITS IN uM and uM/h. 
def VTRPin(b):
  vmax = 700 # Vmax 
  km = 330 # Kilberg p. 169  (effective Km because of other AA)    
  a = vmax*b/(km + b)

  return a
