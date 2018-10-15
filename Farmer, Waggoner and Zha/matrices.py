import numpy as np


# == Parâmetros do modelo original == #

# Coeficientes do modelo

tau    = [0.6137, 0.6137] #
kappa  = [0.6750, 0.6750] #
beta   = [0.9949, 0.9949] #
gamma1 = [2.1900, 0.7700] #
gamma2 = [0.2350, 0.2350] #


# Variância dos choques

rhoD   = [0.7550, 0.7550] #
rhoS   = [0.8350, 0.8350] #
rhoR   = [0.7200, 0.7200] #
sigmaD = [0.2250, 0.2250] #
sigmaS = [0.6206, 0,6206] #
sigmaR = [0.2050, 0.2050] #

# Matriz de transição

P = np.array([[1, 0.0128],
             [0, 0.9872]])

# == Matrizes == #
n = 7
h = 2
s = 2
n_choques = 3

A   = np.array([np.zeros((n,n)) for i in range(h)])
B   = np.array([np.zeros((n,n)) for i in range(h)])
Psi = np.array([np.zeros((n,n_choques)) for i in range(h)])
Pi  = np.array([np.zeros((n,s)) for i in range(h)])

for i in range(h):
    
    A[i][0,0] = 1
    A[i][1,0] = -kappa[i]
    A[i][2,0] = -(1-rhoR[i])*gamma2[i]
    A[i][5,0] = 1
    A[i][0,1] = -1
    A[i][1,2] = 1
    A[i][2,2] = -(1-rhoR[i])*gamma1[i]
    A[i][6,2] = 1
    A[i][0,3] = -tau[i]
    A[i][1,3] = -beta[i]
    A[i][0,4] = tau[i]
    A[i][2,4] = 1
    A[i][0,5] = -1
    A[i][3,5] = 1
    A[i][1,6] = 1
    A[i][4,6] = 1

    B[i][5,1] = 1
    B[i][6,3] = 1
    B[i][2,4] = rhoR[i]
    B[i][3,5] = rhoD[i]
    B[i][4,6] = rhoS[i]

    Psi[i][2,0] = 1
    Psi[i][3,1] = 1
    Psi[i][4,2] = 1
    
    Pi[i][5,0] = 1
    Pi[i][6,1] = 1
