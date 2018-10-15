import numpy as np
from canonical import *
from solve_msdsge import *

def sysmat_msdsge(para):

    ii=0

    kappa  = [para[ii], para[ii]];   ii=ii+1
    phiPI  = [para[ii], para[ii+1]]; ii=ii+2
    phiX   = [para[ii], para[ii+1]]; ii=ii+2
    psiB   = [para[ii], para[ii+1]]; ii=ii+2
    psiG   = [para[ii], para[ii]];   ii=ii+1
    psiR   = [para[ii], para[ii]];   ii=ii+1
    psiPI  = [para[ii], para[ii]];   ii=ii+1
    rho_g  = [para[ii], para[ii]];   ii=ii+1
    rho_z  = [para[ii], para[ii]];   ii=ii+1
    rho_mu = [para[ii], para[ii]];   ii=ii+1

    sigM   = [para[ii], para[ii+1]];   ii=ii+2
    sigF   = [para[ii], para[ii+1]];   ii=ii+2
    sigG   = para[ii];   ii=ii+1
    sigZ   = para[ii];   ii=ii+1
    sigMU  = para[ii];   ii=ii+1

    p00    = para[ii]; ii=ii+1
    p11    = para[ii]; ii=ii+1


    matCan = canonical(para)
    matSol = solve_msdsge(matCan)

    if matSol.RC==1:

        # == Indices for observable equations ==  #

        eq_y   = 0
        eq_pi  = 1
        eq_R   = 2
        eq_tau = 3
        eq_b   = 4

            # == number of observation variables == #

        ny = 5

        # == Variable indices == #

        x_t   = 0
        pi_t  = 1
        R_t   = 2
        tau_t = 3
        b_t   = 4
        g_t   = 5
        z_t   = 6
        mu_t  = 7
        Ex_t1 = 8
        Epi_t1= 9

        # == Shock indices (eps) == #

        R_sh   = 0
        tau_sh = 1
        g_sh   = 2
        z_sh   = 3
        mu_sh  = 4

        #=========================================================================
        #                          TRANSITION EQUATION
        #  
        #           \beta(t) = Phi*\beta(t-1) + R*e(t),    e(t) ~ iid N(0,Se)
        # 
        #=========================================================================

        nep = np.shape(matSol.T0[0])[1]

        Phi = matSol.T1

        R   = matSol.T0

        Se  = np.zeros((2,nep,nep))

        Se[0, R_sh,R_sh]     = (sigM[0])**2
        Se[0, tau_sh,tau_sh] = (sigF[0])**2
        Se[0, g_sh,g_sh]     = (sigG)**2
        Se[0, z_sh,z_sh]     = (sigZ)**2
        Se[0, mu_sh,mu_sh]   = (sigMU)**2

        Se[1, R_sh,R_sh]     = (sigM[1])**2
        Se[1, tau_sh,tau_sh] = (sigF[1])**2
        Se[1, g_sh,g_sh]     = (sigG)**2
        Se[1, z_sh,z_sh]     = (sigZ)**2
        Se[1, mu_sh,mu_sh]   = (sigMU)**2
        
        #=========================================================================
        #                          MEASUREMENT EQUATION
        #  
        #           y(t) = C + B*s(t) + u(t),    u(t) ~ N(0,HH)          
        # 
        #=========================================================================

        C = np.zeros((ny,1))

        nstate = np.shape(Phi)[1] 

        B = np.zeros((ny,nstate))
        B[eq_y, eq_y]     = 1
        B[eq_pi, eq_pi]   = 1
        B[eq_R, eq_R]     = 1
        B[eq_tau, eq_tau] = 1
        B[eq_b, eq_b]     = 1

        H = np.zeros((ny,ny))
    

    class Sysmat:
        
        def __init__(self, C, B, H, R, Se, Phi):
            
            self.C   = C
            self.B   = B
            self.H   = H
            self.R   = R
            self.Se  = Se
            self.Phi = Phi
            
    matrices = Sysmat(C, B, H, R, Se, Phi)
    
    return matrices            