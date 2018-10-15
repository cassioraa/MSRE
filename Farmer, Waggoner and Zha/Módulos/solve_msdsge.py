def solve_msdsge(matCan):
    
    r'''Description: This function take an object containing the matrices of canonical form of a MS-DSGE
    and solves this model using the method described in Farmer, Waggoner and Zha (2011)
    
    Input: matCan, an object containing the matrices of canonical form. See canonical.py
    Output: solution, an object containing the matrices of the following solution:

    x_t = T1 x_{t-1} + T0 \epsilon_t

    RC = -1, No MSV solution
    RC = 0, MSV solution is not Mean Square Stable
    RC = 0, MSV solution is Mean Square Stable '''

    #============================================
    # Some options for the solution algorithm
    #============================================

    # number of forward-look variables
    l = np.shape(matCan.PPI[1])[1] 

    # maximum number of iterations
    maxit = 100

    # convergence criterion
    smallval = 1.0e-10

    # number of endogenous variables
    n = np.shape(matCan.A[0])[0]

    # number of exogenous variables
    nsh = np.shape(matCan.PSI[0])[1]

    # number of policies regimes
    h_p_regimes = np.shape(matCan.P)[0]

    #============================================
    # Solving the model
    #============================================


    matSol = fwz_msv_msre_singular(P=matCan.P,
                                   A=matCan.A,
                                   B=matCan.B,
                                   Psi=matCan.PSI,
                                   s=l,
                                   z=None, 
                                   max_count=maxit,
                                   tol=smallval)

    # In case the algorithm has not achieve convergence

    if matSol.err < 0:
        print("sdf")
        dim = np.shape(matSol.z)[0]
        z00 = np.random.normal(0,1,dim).reshape((dim,1))/100

        matSol = fwz_msv_msre_singular(P=matCan.P,
                                   A=matCan.A,
                                   B=matCan.B,
                                   Psi=matCan.PSI,
                                   s=l,
                                   z=z00, 
                                   max_count=maxit,
                                   tol=smallval)

    if matSol.err < 0:
        print(matSol.err)
        RC = -1 # No MSV
        T1 = 0
        T0 = 1

    else:

        T1 = np.zeros((h_p_regimes, n, n)) 
        T0 = np.zeros((h_p_regimes, n, nsh)) # impact

        for i in range(h_p_regimes):

            T1[i,:,:] = np.dot(matSol.V[i], matSol.F1[i])
            T0[i,:,:] = np.dot(matSol.V[i], matSol.G1[i])


        #====== Checking stationarity of the solution. =======
        # Checking the stationarity of regime-switching models. 
        # If the rows of P sum to one and Gamma(i)=V(i)*A(i) is ngensys x ngensys,
        # then the solution is MSS stable if and only if the eigenvalues of
        #      kron(P',eye(ngensys*ngensys)) *diag(kron(Gamma(i),Gamma(i)))
        # are all inside the unit circle.

        stackblkdiag = np.zeros((h_p_regimes*n**2,h_p_regimes*n**2))

        for i in range(h_p_regimes):
            stackblkdiag[(i*n**2):(i+1)*n**2, (i*n**2):(i+1)*n**2] = np.kron(T1[i,:,:], T1[i,:,:])

        stackmat = np.dot((np.kron(matCan.P, np.eye(n*n))), stackblkdiag)
        autovalores, autovetores = np.linalg.eig(stackmat)

        if max(np.abs(autovalores)) < 1:
            RC = 1 # MSV solution is MSS
        else:
            RC = 0 # MSV solution is not MSS
            
        
        class Solution:
            
            def __init__(self,T1, T0, RC, z):
                self.T1 = T1
                self.T0 = T0
                self.RC = RC
                self.z  = matSol.z
    
    solution = Solution(T1, T0, RC, z)
    
    return solution