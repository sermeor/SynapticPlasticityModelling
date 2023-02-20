## Function of rate of inhibition of histidine decarboxylase (HTDC) by FMH suicide inhibition.
# b is concentration of FMH in the brain (uM), a is the rate of inhibition
# (h-1).
def k_fmh_inh(b):
  k_fmh = 10.4040 #Constant of FMH suicide inhibition reaction (h-1).
  ki = 8.3 #FMH dissociation constant in uM. 
  a = k_fmh * b/(ki + b)
  return a