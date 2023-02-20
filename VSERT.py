# Rate of synthesis of 5-HTP from tryptophan with bh4 as co-substract. 
# b = tryptophan
# d = BH4
# UNITS IN uM and uM/h. 

def VTPH(b, d):
  k1 = 40 # Km for tryptophan Haavik05 (homo sapiens) (rats in same range)               
  k2 = 20 # Km for BH4 Haavik05
  k4 = 1000 # (Ki is 970 in Haavik05)
  k5 = 278 # Vmax mult by two when we halved inhib term at steady state
  
   
  #a = ((k5.*b)./(k1 + b +b.^2./k4)).*(c./(k2 + c)); %no auto
  #a = ((k5.*b)./(k1 + b +b.^2./k4)).*(c./(k2 + c)).*(1.5 - e.^2./((.000766).^2 + e.^2)); % auto
  # In Haavik 05 the SI curve peaks at about 150. In Friedman72 it peaks at
  # about 200. 
  
  a = (k5*b)/(k1 + b + b^2/k4)*(d/(k2 + d));
  
  return a