# Uptake 2 rate of reuptake into glial terminals. 
# b = e5ht
# UNITS IN uM and uM/h. 

def VUP2(b):
  km = 0.17 #Km Wightman-bunin
  vmax = 1400 #Vmax Wightman-Bunin .78*1800
  a = (vmax*b/(km + b))  # (4700)
  # Km is from Bunin98, see also Daws05
  
  return a