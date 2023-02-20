# Function of monoamine transporter in vesicles.
# Transport of histamine from cytosol to vesicles.
# b = cha
# c = vha
def VMATHmc(b, c):
  k = 24     
  V =  21104
  a = (V*(b/(k + b)) - 5*c)
  if a<0:
    a = 0 #Make sure is not negative. 

  return a