# Function of transport from vht reserve to readily vht. 
# vht is the current value of vesicular serotonin
# vht_basal is the steady state value. 
def vht_trafficking(vht, vht_basal):
  max_r = 100; 
  min_r = 0;
  s = 15; #Strength
  diff = vht_basal - vht; #Difference respect to basal. 
  ratio = s*diff;

  if ratio > max_r:     
    ratio = max_r   
  elif ratio < min_r:     
    ratio = min_r

  return ratio