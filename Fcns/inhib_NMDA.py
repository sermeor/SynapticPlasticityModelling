##Function of occupacy ratio of NMDA receptors depending on ketamine and norketamine concentration (uM)
#This function assumes that k and nk don't inhibit each other, and that glutamate concentration has no effect on binding rate of k and nk, since their binding stregth is much greater. 

def inhib_NMDA(k, nk):
    #Hill equation (MM non-competitive inhibition)
    #Not affected by glutamate concentration.
    n_k = 1.4 #Hill number ketamine. 
    n_nk = 1.2 #Hill number nor-ketamine.
    Ki_k = 2 #Ketamine concentration for half occupacy (uM)
    Ki_nk = 17 #Norketamine concentration for half occupacy (uM)
    f = 1/(1 + (Ki_k/k)**n_k) + 1/(1 + (Ki_nk/nk)**n_nk)
    return f