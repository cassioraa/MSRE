import numpy as np
import scipy as sp
from scipy import linalg
import pandas as pd




def fwz_msv_msre(P, A, B, Psi, s,x=None, max_count=100, tol=1e-5):
    '''
    Descrição da Função
    
    Inputs:
    
    P: Uma matriz h x h, em que h é o número de regimes 
    A: Um array contendo h matrizes n x n
    B: Um array contendo h matrizes n x n
    Psi: um array contendo h matrizes
    s: Número de erros expectacionais (corresponde ao l do artigo)
    x: Chute inicial (default = 0 vetor)
    max_count: Número máximo de iterações (default = 1000)
    tol: Critério de convergência (default = 1e-5)
    
    '''

    # == Alocar espaço para salvar os resulados == #
    
    h = np.size(P, 0) # número de linhas de P
    n = np.size(A[1], 0) # número de linhas de A
    r = np.size(Psi[1], 1) #número de colunas de Psi
    
    if x is None:
        x = np.zeros((h*s*(n-s), 1)) # Cria um vetor de zeros (chute inicial) 
    
    f     = np.zeros((h*s*(n-s), 1)) # Cria vetor de zeros 
    Is    = np.eye(s)

    In_s  = np.eye((n-s))
    
    U = np.array([np.zeros((n,n)) for j in range(h)])
    
    for i in range(h):
        U[i] = np.linalg.solve(A[i], np.eye(len(A[i])))
    
    C = np.array([[np.zeros((n,n)) for i in range(h)] for j in range(h)])
    
    for i in range(h):
        for j in range(h):
            C[i][j] = (P[i,j]*np.dot(B[j], U[i]))
            
    cout  = True
    count = 1
    D     = np.zeros((h*s*(n-s),h*s*(n-s))) # Cria uma matriz (hs(n-s) x hs(n-s)) para armazenar os resultados de f'
    X     = np.array([np.zeros((s, n-s)) for i in range(h)]) # Cria uma h matriz (n x n) para armazenar cada chute inicial
    
    for i in range(h):
        X[i] = x[i*s*(n-s):(i+1)*s*(n-s)].reshape((s, n-s))
        
    # == Abre o while == #
    while cout:
        for i in range(h):
            for j in range(h):
                W1 = (np.dot(C[i][j], np.concatenate((In_s, -X[i]),0)))
                W2 = (W1[0:n-s,:]) # Seleciona as n-s primeiras linhas de W1 (=multiplicar por [I 0])
                D[i*s*(n-s):(i+1)*s*(n-s), j*s*(n-s):(j+1)*s*(n-s)] = (np.kron(W2.transpose(), Is))
                if i==j:
                    W1 = np.zeros((s,n))
                    for k in range(h):
                        W1 = (W1 + np.dot(np.concatenate((X[k], Is), 1), C[i][k]))
                    W2 = (-W1[:, n-s:n])
                    D[i*s*(n-s):(i+1)*s*(n-s), j*s*(n-s):(j+1)*s*(n-s)] = (D[i*s*(n-s):(i+1)*s*(n-s), j*s*(n-s):(j+1)*s*(n-s)] + np.kron(In_s,W2))
        
        for i in range(h):
            mf = np.zeros((s,n-s))
            for j in range(h):
                mf = (mf + np.dot(np.dot(np.concatenate((X[j], Is), 1), C[i][j]), np.concatenate((In_s, -X[i]))))
            f[i*s*(n-s):(i+1)*s*(n-s)] = (mf.T.reshape((s*(n-s),1)))
        
        # Resolve o sistema y = D^{-1}f

        
        if np.shape(D)[0] == np.shape(D)[1]:
            y = np.linalg.solve(D, f)
        else:
            y = np.linalg.lstsq(D,f)

        x = x - y
        
        if count > max_count or np.linalg.norm(f) < tol:
            cout = False
        else:
            count = count + 1
            for i in range(h):
                X[i] = x[(i)*s*(n-s):(i+1)*s*(n-s)].reshape((s, n-s), order="F")

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
          
    return [F1, F2, G1, G2, V, err,x]   