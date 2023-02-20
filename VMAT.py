# Vesicular monoamine transporter, 5ht -> v5ht.
# b = cDA
#c= vda
#UNITS IN uM and uM/h. 
def VMAT(b,c):
  k = 0.2 #was 1 before Nov 3.
  V = 1230 #1230 was 6300 before Nov 3  Vmax = 3500 in paper
  a = (V*b/(k + b))-(1)*c; 
  # Km for uptake by vesicles = 123 or 252 nM (Slotkin77)
  # Km for uptake by vesicles =  198 nM (Rau06)

  if a < 0:
    a = 0 #Make sure a is not negative. 

  return a
