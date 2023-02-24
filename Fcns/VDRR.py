## Rate of production of bh4 from bh2. 
# UNITS in uM and uM/h
# b = BH2
# c = NADPH
# d = BH4
# e = NADP
def VDRR(b,c,d,e):
  k1 = 100   #Km for BH2 (BRENDA) (6-650)
  k2 = 75  #Km for NADPH (BRENDA, values 770,110,29) (schumber 70-80)
  V1 = 5000 #Vmax forward
  k3 = 10  #Km for BH4 (BRENDA) (1.1 to 17)
  k4 = 75  #Km for NADP (BRENDA)(schumber 70-80)
  V2 = 3  #Vmax backward
  
  # forward direction from BH2 to BH4
  a = V1*b*c/((k1 + b)*(k2 + c))  - V2*d*e/((k3 + d)*(k4 + e))
  return a
