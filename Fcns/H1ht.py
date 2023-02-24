## Factor of uptake 2 velocity (Vu2) depending on eht (a)
#and basal eht (c)
# UNITS IN uM.
def H1ht(a, c):
  if a < 19:
    f = 0
  elif  a < (c + 0.02):
    f = (50)*(a-c)
  else:
    f = 1
  return f