## Function of monoamine transporter in vesicles.
# Transport of histamine from cytosol to vesicles
# b = cha
# c = vha
def VMATH(b, c):
  km = 24        
  vmax =  10552
  a = (vmax*(b/(km + b)) - 5*c)
  if a<0:
    a = 0 #Make sure is not negative. 
  return a