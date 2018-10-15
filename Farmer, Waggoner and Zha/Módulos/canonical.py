def canonical(para):
    
    r'''Description: This function takes a set of parameter and construct a system of 
    markov switching rational expectation in the following form:
    
    A(s_t)x_t = B(s_t)x_{t-1} + \Psi(s_t)\epsilon_t + \Pi(s_t)\eta_t
    
    Input: An 1-dimensional array
    Output: A(s_t), B(s_t), \Psi(s_t), \Pi(s_t) and the transition matrix P of the states s_t'''   
    

    kappa  = [para[0], para[0]]
    phiPI  = [para[1], para[2]]
    phiX   = [para[3], para[4]]
    psiB   = [para[5], para[6]]
    psiG   = [para[7], para[7]]
    psiR   = [para[8], para[8]]
    psiPI  = [para[9], para[9]]
    rho_g  = [para[10], para[10]]
    rho_z  = [para[11], para[11]]
    rho_mu = [para[12], para[12]]
    sigM   = [para[13], para[13]]
    sigF   = [para[14], para[14]]
    sigG   = [para[15], para[15]]
    sigZ   = [para[16], para[16]]
    sigMU  = [para[17], para[17]]
    p00    = para[18]
    p11    = para[19]


    # == Calibrated parameters == #
    sigma = [1,1]
    beta  = [0.998, 0.998]
    bbar  = [0.20, 0.2] 

    # Transition matrix

    P = np.array([[p00[0], 1-p11[0]],
                  [1-p00[0], p11[0]]])

    #=========================================================================
    #                       DEFINE OBJECTS
    #=========================================================================

    # Equation indices

    eq_1 = 0  # Dynamic IS
    eq_2 = 1  # Philipps curve
    eq_3 = 2  # Taylor rule
    eq_4 = 3  # Fiscal rule
    eq_5 = 4  # Debt dynamic
    eq_6 = 5  # Exogenous process for g 
    eq_7 = 6  # Exogenous process for z
    eq_8 = 7  # Exogenous process for mu
    eq_9 = 8  # y_{t+1}
    eq_10= 9 # pi_{t+1}

    # Variable indices 

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

    # Number of policy regimes

    h_regimes = 2

    # Expectation error indices (eta) 

    ex_sh  = 0
    epi_sh = 1

    # Shock indices (eps)

    R_sh   = 0
    tau_sh = 1
    g_sh   = 2
    z_sh   = 3
    mu_sh  = 4

    # SUMMARY

    neq  = 10
    neta = 2
    neps = 5

    # == initialize matrices == #

    A   = np.zeros((h_regimes, neq, neq))
    B   = np.zeros((h_regimes, neq, neq))
    PSI = np.zeros((h_regimes, neq,neps))
    PPI = np.zeros((h_regimes, neq,neta))

        #=========================================================================
        #                EQUILIBRIUM CONDITIONS: CANONICAL SYSTEM
        #=========================================================================


    for i in range(h_regimes):

        #=========================================================================
        #         1. 
        #=========================================================================

        A[i][eq_1,x_t]     =  1
        A[i][eq_1,R_t]     =  sigma[i]
        A[i][eq_1,z_t]     = -1
        A[i][eq_1, Ex_t1]  = -1
        A[i][eq_1, Epi_t1] = -sigma[i]

        #=========================================================================
        #         2. 
        #=========================================================================

        A[i][eq_2,x_t]     = -kappa[i]
        A[i][eq_2,pi_t]    = 1
        A[i][eq_2,mu_t]    = -1
        A[i][eq_2, Epi_t1] = -beta[i]


        #=========================================================================
        #         3. 
        #=========================================================================

        A[i][eq_3,x_t]  = -phiX[i]
        A[i][eq_3,pi_t] = -phiPI[i]
        A[i][eq_3,R_t]  = 1
        PSI[i][eq_3,R_sh]  = 1

        #=========================================================================
        #         4. 
        #=========================================================================

        A[i][eq_4,tau_t] = 1
        A[i][eq_4,g_t]   = -psiG[i]
        A[i][eq_4,R_t]   = -psiR[i]
        A[i][eq_4,pi_t]  = -psiPI[i]
        B[i][eq_4,b_t]   = psiB[i]
        PSI[i][eq_4,tau_sh] = 1

        #=========================================================================
        #         5. 
        #=========================================================================

        A[i][eq_5, pi_t]  = (1/beta[i])*bbar[i]
        A[i][eq_5, R_t]   = -bbar[i]
        A[i][eq_5, tau_t] = (1/beta[i])
        A[i][eq_5, b_t]   = 1
        A[i][eq_5, g_t]   = -(1/beta[i])
        B[i][eq_5, b_t]   = (1/beta[i])

        #=========================================================================
        #         6. 
        #=========================================================================

        A[i][eq_6,g_t]     = 1
        B[i][eq_6, g_t]    = rho_g[i]
        PSI[i][eq_6, g_sh] = 1

        #=========================================================================
        #         7. 
        #=========================================================================

        A[i][eq_7,z_t]     = 1
        B[i][eq_7, z_t]    = rho_z[i]
        PSI[i][eq_7, z_sh] = 1

        #=========================================================================
        #         8. 
        #=========================================================================

        A[i][eq_8, mu_t]    = 1
        B[i][eq_8, mu_t]    = rho_mu[i]
        PSI[i][eq_8, mu_sh] = 1

        #=========================================================================
        #         9. 
        #=========================================================================

        A[i][eq_9, x_t]     = 1
        B[i][eq_9, Ex_t1]   = 1
        PPI[i][eq_9, ex_sh] = 1

        #=========================================================================
        #         10. 
        #=========================================================================

        A[i][eq_10, pi_t]     = 1
        B[i][eq_10, Epi_t1]   = 1
        PPI[i][eq_10, epi_sh] = 1
    
    class Matrices:
        
        def __init__(self, A, B, PSI, PPI, P):
            
            self.A = A
            self.B = B
            self.PSI = PSI
            self.PPI = PPI
            self.P = P
    
    results = Matrices(A,B,PSI,PPI,P)
    
    return results