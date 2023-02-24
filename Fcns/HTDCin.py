## Function of replenishment of histidine decarboxylase (HDC) during FMH inhibition. 
# b represents ratio levels of HDC, while a represents rate of
# replenishment (h-1).

def HTDCin(b):
  s = 0.55 #Strength of replenishment. 
  a = s*(1 - b)
  return a