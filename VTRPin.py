# tryptophan transporter rate from blood to cytosol. 
# b = btyr
# UNITS IN uM and uM/h. 
def VTRPin(b):
  vmax = 700 # Vmax 
  km = 330 # Kilberg p. 169  (effective Km because of other AA)    
  a = vmax*b/(km + b)

  return a
  
  #Partridge75 says Km = 190 muM (with respect to total trp)
  #Smith87   says Km = 15 muM  (with respect to free trp)   
  #(both in BBB folder)
  
  # kilber flux in to brain is 157 muM/hr 2.61/min  page 169
  # kilber tyr flux in to brain is  muM/hr 4.14/min  page 169