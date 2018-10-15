def fwz_msv_msre_singular(P, A, B, Psi, s, z, max_count, tol):

    r'''Description: This function implement the algorithm describe in Farmer, Waggoner and Zha (2011) `MINIMAL STATE VARIABLE SOLUTIONS TO
        MARKOV-SWITCHING RATIONAL EXPECTATIONS MODELS'

    '''
    
    # == Alocar espaço para salvar os resulados == #

    h = np.size(P, 0) # número de linhas de P
    n = np.size(A[1], 0) # número de linhas de A
    r = np.size(Psi[1], 1) #número de colunas de Psi

    if z is None:
        z = np.zeros((h*s*(n-s), 1)) # Cria um vetor de zeros (chute inicial) 
    #x = None
    #if x is None:
    #    x = np.zeros((h*s*(n-s), 1)) # Cria um vetor de zeros (chute inicial) 

    f     = np.zeros((h*s*(n-s), 1)) # Cria vetor de zeros 
    Is    = np.eye(s)

    In_s  = np.eye((n-s))

    U = np.zeros((h,n,n))
    R = np.zeros((h,n,n)) 
    Q = np.zeros((h,n,n)) 

    C1 = np.zeros((h,s,n-s))
    C2 = np.zeros((h,s,s))



    for i in range(h):
        Q[i], R[i] = np.linalg.qr(A[i].T)

        Ui0  = np.hstack((np.linalg.inv(R[i][0:n-s,0:n-s].T), np.zeros((n-s,s))))
        Ui1  = np.hstack((np.zeros((s,n-s)), np.eye(s)))
        RR   = np.vstack((Ui0,Ui1))

        U[i] = np.dot(Q[i], RR)

        AiUi  = np.dot(A[i],U[i])
        C1[i] = AiUi[n-s:n+1,0:n-s]
        C2[i] = AiUi[n-s:n+1,n-s:n+1]

    C = np.zeros((h,h,n,n))

    for i in range(h):
        for j in range(h):
            C[i][j] = (P[i,j]*np.dot(B[j], U[i]))

    cout  = True
    count = 1
    D     = np.zeros((h*s*(n-s),h*s*(n-s))) # Cria uma matriz (hs(n-s) x hs(n-s)) para armazenar os resultados de f'
    X     = np.zeros((h,s,n-s)) # Cria uma h matriz (n x n) para armazenar cada chute inicial
    Z     = np.zeros((h,s,n-s)) # Cria uma h matriz (n x n) para armazenar cada chute inicial

    for i in range(h):
        Z[i] = z[i*s*(n-s):(i+1)*s*(n-s)].reshape((s, n-s), order="F")
        X[i] = np.dot(C2[i],Z[i]) - C1[i]

    while cout:
        for i in range(h):

            mf = np.zeros((s,n-s))

            for j in range(h):

                mf = mf + np.dot(np.dot(np.hstack((X[j], Is)),C[i][j]),
                                 np.vstack((In_s, -Z[i])))

            f[i*s*(n-s):(i+1)*s*(n-s)] = (mf.T.reshape((s*(n-s),1)))

        if count > max_count or np.linalg.norm(f) < tol:
            cout = False
        else:

            for i in range(h):
                for j in range(h):
                    W1 = (np.dot(C[i][j], np.concatenate((In_s, -Z[i]),0)))
                    W2 = (W1[0:n-s,:]) # Seleciona as n-s primeiras linhas de W1 (=multiplicar por [I 0])
                    D[i*s*(n-s):(i+1)*s*(n-s), j*s*(n-s):(j+1)*s*(n-s)] = (np.kron(W2.transpose(), C2[j]))

                    if i==j:
                        W1 = np.zeros((s,n))
                        for k in range(h):
                            W1 = (W1 + np.dot(np.concatenate((X[k], Is), 1), C[i][k]))
                        W2 = (-W1[:, n-s:n])
                        D[i*s*(n-s):(i+1)*s*(n-s), j*s*(n-s):(j+1)*s*(n-s)] = (D[i*s*(n-s):(i+1)*s*(n-s), j*s*(n-s):(j+1)*s*(n-s)] + np.kron(In_s,W2))

            # Resolve o sistema y = D^{-1}f


            if np.shape(D)[0] == np.shape(D)[1]:
                y = np.linalg.solve(D, f)
            else:
                y = np.linalg.lstsq(D,f)


            z = z - y

            count = count + 1
            for i in range(h):
                Z[i] = z[i*s*(n-s):(i+1)*s*(n-s)].reshape((s, n-s), order="F")
                X[i] = np.dot(C2[i],Z[i]) - C1[i]
    # == Fecha o while == #

        if np.linalg.norm(f) < tol:
            err = count
        else:
            err = -count

    F1 = np.array([np.zeros((n-s,n)) for j in range(h)])
    F2 = np.array([np.zeros((s,n)) for j in range(h)])
    G1 = np.array([np.zeros((n-s,r)) for j in range(h)])
    G2 = np.array([np.zeros((s,r)) for j in range(h)])
    V  = np.array([np.zeros((n,n-s)) for j in range(h)])
    pi = np.concatenate((np.zeros((n-s, s)), Is), 0)

    for i in range(h):
        X       = (x[i*s*(n-s):(i+1)*s*(n-s)].reshape((s,n-s), order="F"))
        V[i]    = (np.dot(U[i], np.concatenate((In_s, -X), 0)))
        W       = (np.concatenate((np.dot(A[i],V[i]), pi), 1))

        if np.shape(W)[0] == np.shape(W)[1]:
            F = np.linalg.solve(W,B[i])
        else:
            F = np.linalg.lstsq(W,B[i])


        F1[i] = (F[0:n-s, :])
        F2[i] = (F[n-s:n+1, :])

        if np.shape(W)[0] == np.shape(W)[1]:
            G = np.linalg.solve(W,Psi[i])
        else:
            G = np.linalg.lstsq(W,Psi[i])

        G1[i] = (G[0:n-s, :])
        G2[i] = (G[n-s:n+1, :])
        
    return (F1, F2, G1, G2, V, err, z)