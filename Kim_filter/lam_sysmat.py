import numpy as np


def sys_mat(params, y):
    
    """ DESCRIPTION
   The general form for a state-space model with regime switching 
   (following the notation of Kim and Nelson (1999, pg. 98)) is the 
   following:

   Measurement equation:
   y_t = H(s_t)*x_t + B(s_t)*z_t + e_t           e_t ~ N(0, R(s_t))

   Transition equation:
   x_t = mu(s_t) F(s_t)*x_{t-1} + G(s_t)*v_t    v_t ~ N(0, Q(s_t))

   Transitional probability matrix:
   p = [p11, p21, ... , pM1; p12, ..., pM2; ... ; p1M, ... pMM]; where pjk
   = Pr{S_t=k|S_(t-1)=j}

   The Lam Model of output growth is represented (in abbreviated form) as:
   the following in Kim and Nelson (1999, pgs. 111-112).

   Measurement equation:
   d_y_t = [1 1]*[x_t x_(t-1)]' + delta_j
   where d_y_t, is the growth rate of gdp, x_t represents the stationary
   cycle component of gdp, and delta represents the markov-switching mean
   growth rate of gdp

   Transition equation:
   [x_t x_(t-1)]' = [theta_1 theta_2; 1, 0]*[x_(t-1) x_(t-2)]'+[u_t, 0]'
   where [u_t 0]' ~ N([0, 0]',[s2, 0; 0, 0])"""
    
    #=========================================================================#
    # unpacking parameters
    #=========================================================================#
    p    = params[0]
    q    = params[1]
    d1   = params[2]
    d2   = d1 + params[3]
    sig  = params[4]
    phi1 = params[5]
    phi2 = params[6]
    #b11  = params[7]
    #b12  = params[8]

    #=========================================================================#
    # Set up the state-space form with hyperparamters
    #=========================================================================#

    H1 = np.array([[1,-1]]) #
    H2 = np.array([[1,-1]]) #
    H  = np.array([H1, H2]) #

    A1 = np.array([d1]) #
    A2 = np.array([d2]) #
    A  = np.array([[A1], [A2]]) #

    R1  = np.array([[0]]) #
    R2  = np.array([[0]]) #
    R   = np.array([R1, R2]) #

    mu1 = np.array([[0]])
    mu2 = np.array([[0]])
    mu  = np.array([mu1, mu2])

    F1  = np.array([[phi1, phi2], [1,0]]) #
    F2  = np.array([[phi1, phi2], [1,0]]) #
    F   = np.array([F1, F2])

    G1  = np.eye(2) #
    G2  = np.eye(2) #
    G   = np.array([G1, G2]) #

    Q1  = np.array([[sig**2, 0], [0,0]]) #
    Q2  = np.array([[sig**2, 0], [0,0]]) #
    Q   = np.array([Q1, Q2])

    prob   = np.array([[q, 1-p],[1-q,p]] ) #TODO: check if the columns add to 1
    
    T = len(y)
    z = np.ones((T,1))
    
    
    # Initial values
    J = np.shape(F[1])[1] # number of unobserved vector beta
    
    P01 = np.reshape(np.dot(np.linalg.inv(np.eye(J**2) -np.kron(F[0], F[0])), np.reshape(Q[0],(J**2,1))),(J,J))
    P02 = P01
    P0  = np.array([P01, P02])
    
    #b01 = np.array([[b11],[b12]])
    b01 = np.zeros((2,1))
    b02 = b01
    b0  = np.array([np.array([[5.224],[0.535]]),
                    np.array([[5.224],[0.535]])])

    #b0 = np.array([b01,b02])
    
    # Initial probabilities [eq.(4.49) Kim and Nelson]
    AA   = np.concatenate((np.eye(J)-prob, np.ones((1,J))),0)
    E    = np.concatenate((np.zeros((J,1)), np.array([[1]])),0)
    S0  = np.dot(np.linalg.inv(np.dot(AA.T, AA)), np.dot(AA.T,E)) # eq. (22.2.26) Hamilton (1999)
    
    class Results:
        
        def __init__(self, H, A, R, mu, F, G, Q, prob, P0, S0, b0, z, y):
            
            self.H = H
            self.A = A
            self.R = R
            self.mu = mu
            self.F = F
            self.G = G
            self.Q = Q
            self.prob = prob
            self.P0 = P0
            self.S0 = S0
            self.b0 = b0
            self.z = z
            self.y = y
            
    results = Results(H, A, R, mu, F, G, Q, prob, P0, S0, b0, z, y)
        
    return results