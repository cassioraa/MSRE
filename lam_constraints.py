import numpy as np

def lam_constraints(params):
    
    """Description:"""
    params_constrained = np.zeros((len(params),1))

    # Probabilities (p and q)

    params_constrained[0] = np.exp(params[0])/(1+np.exp(params[0]))
    params_constrained[1] = np.exp(params[1])/(1+np.exp(params[1]))

    # No constraints on delta_1 and delta_2

    params_constrained[2] = params[2]
    params_constrained[3] = params[3]

    # No constraints on sigma>0 (because it is squared in the model)

    params_constrained[4] = params[4]

    # Constrains so that phi_1 and phi_2 jointly satisfy stationarity properties (see Hamilton (1994))

    tmp1 = params[5]/(1+np.abs(params[5]))
    tmp2 = params[6]/(1+np.abs(params[6]))

    params_constrained[5] = tmp1+tmp2
    params_constrained[6] = -tmp1*tmp2
    
    #params_constrained[7] = params[7]
    #params_constrained[8] = params[8]

    return params_constrained