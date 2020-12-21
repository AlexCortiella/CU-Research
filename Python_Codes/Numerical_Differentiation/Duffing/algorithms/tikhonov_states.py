# IMPORTS
import numpy as np
from scipy.sparse import coo_matrix, eye
import scipy.linalg as sla
from scipy.linalg import norm
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from scipy import interpolate
import statsmodels.api as sm

#FUNCTION DEFINITIONS

#Local function to compute the difference matrices of any order
def diff_mat(n,order):
    #Zero'th derivative
    if order==0:
        D = eye(n)
    else:
        #Compute D of specific order
        c = np.array([-1,1] + [0] * (order-1))

        nd = n-order
        for i in range(1,order):
            c = np.append(0,c[0:order]) - np.append(c[0:order],0)

        D = coo_matrix((nd, n), dtype=np.int8)

        for i in range(0,order+1):
            row = np.array(range(nd))
            col = np.array(range(nd)) + i
            val = c[i] * np.ones(nd)
            D = D + coo_matrix((val, (row, col)), shape=(nd, n))

    return D

def menger(P1, P2, P3):
    
    u = P1 - P2
    v = P3 - P2

    return 2 * (v[0] * u[1] - u[0] * v[1]) / ( norm(u) * norm(v) * norm(u - v) )

def check_lambdas(lambda_min, lambda_max, y, tol = 1e-8):
    
    #Make sure that the solver does not fail for the range of lambdas 
#     while True:
#         try:
#             out = trend_filter(y, lambd = lambda_min, order = order)
#             beta_min = out[0]
#             residual_min, reg_residual_max = out[1]
#         except:
#             print('lambda_min too small. Increasing it 10 times...')
#             lambda_min = lambda_min * 10
#             continue
#         else:
#             break
    
#     while True:
#         try:
#             out = trend_filter(y, lambd = lambda_max, order = order)
#             beta_max = out[0]
#             residual_max, reg_residual_min = out[1]
#         except:
#             print('lambda_max too large. Reducing it 10 times...')
#             lambda_max = lambda_max / 10
#             continue
#         else:
#             break
    
    out = tikhonov(y, lambd = lambda_min)
    residual_min, reg_residual_max = out[1]
    out = tikhonov(y, lambd = lambda_max)
    residual_max, reg_residual_min = out[1]
    
    #Make sure the residuals are not too small to help lcurve_corner converge
    while residual_min < tol:
        print('lambda_min too small. Increasing it 10 times...')
        lambda_min = lambda_min * 10
        out = tikhonov(y, lambd = lambda_min)
        residual_min, reg_residual_max = out[1]

    while reg_residual_min < tol:
        print('lambda_max too large. Reducing it 10 times...')
        lambda_max = lambda_max / 10
        out = tikhonov(y, lambd = lambda_max)
        residual_max, reg_residual_min = out[1]
            
    return (lambda_max, lambda_min)

def normalize_get(residual, reg_residual):
    
    residual_max = np.max(residual)
    reg_residual_max = np.max(reg_residual)
    
    residual_min = np.min(residual)
    reg_residual_min = np.min(reg_residual)
    
    #Compute normalization coefficients to transform the L-curve into a [-1, 1] range
    cres0 = 1 - 2/(np.log10(residual_max) - np.log10(residual_min)) * np.log10(residual_max)
    cres1 = 2 / (np.log10(residual_max) - np.log10(residual_min))
    creg0 = 1 - 2/(np.log10(reg_residual_max) - np.log10(reg_residual_min)) * np.log10(reg_residual_max)
    creg1 = 2 / (np.log10(reg_residual_max) - np.log10(reg_residual_min))
    
    
    return (cres0, cres1, creg0, creg1)

def normalize_fit(residual, reg_residual, normalization_coefs):
    
    #Unpack coefficients
    cres0, cres1, creg0, creg1 = normalization_coefs
    
    
    xi = cres0 + cres1 * np.log10(residual)
    eta = creg0 + creg1 * np.log10(reg_residual)
    
    return (xi, eta)

def tikhonov(y, lambd = 0):
    
    """
    hp_filter(y, lambda = 0, order = 3)
    
    finds the solution of the l2 trend estimation problem
    
     minimize    (1/2)||y-f(x)||^2 + lambda*||D^2 f(x)||_2^2,
    
    with variable x, and problem data y and lambda, with lambda >0.
    
    Input arguments:
    
     - y:          n-vector; original signal, dependent variable y(x)
     - x:          n-vector; independent variable
     - lambda:     scalar; positive regularization parameter
    
    Output arguments:
    
     list[0]
     - y_tf:          n-vector; filtered solution
     - dy_tf:         n-vector; filtered derivative
     
     list[1]
     - residual:      l-2 norm of (y - f(x))
     - reg_residual:  l-2 norm of D^2 f(x)
    
    Author: Alexandre Cortiella
    Affiliation: University of Colorado Boulder
    Department: Aerospace Engineering Sciences
    Date: 11/09/2020
    Version: v1.0
    Updated: 11/09/2020
    
    """
    cycle, y_tik = sm.tsa.filters.hpfilter(y, lambd)
    
    # DIMENSIONS
    m   = len(y) #length of signal x

    # OPERATOR MATRICES
    D = diff_mat(m,2)
    
    #Compute residuals
    residual = norm(y - y_tik)
    reg_residual = norm(D * y_tik, ord = 2)
    
    #Generate Tikhonov matrix
    I = eye(m)
    T = (I + lambd * D.T * D)
    
    #Compute the GCV funtion
    S = spsolve(T, I)
    df = np.trace(S.todense())
    GCV = (m * norm(y_tik - y)**2) / (m - df) ** 2
   
    return [y_tik, (residual, reg_residual), (GCV, df)]

def lcurve_corner(y, lambda_min = 1e-10, lambda_max = 1e10, epsilon = 0.001, max_iter = 50, normalize = False, verbose = False):
    
    n = len(y)
    pGS = (1 + np.sqrt(5))/2 #Golden search parameter
    gap = 1
    itr = 0
    lambda_itr = []

    #Check the range of lambdas and compute normalization coefficients
    #lambda_max, lambda_min = check_lambdas(lambda_min, lambda_max, y)

    lambda_vec = np.array([lambda_min, lambda_max, 0, 0])

    lambda_vec[2] = 10 ** ( (np.log10(lambda_vec[1]) + pGS * np.log10(lambda_vec[0])) / (1 + pGS) ) 
    lambda_vec[3] = 10 ** ( np.log10(lambda_vec[0]) + np.log10(lambda_vec[1]) - np.log10(lambda_vec[2]) )

    ress = np.zeros(4)#residuals
    regs = np.zeros(4)#regularization residuals

    while (gap >= epsilon) and (itr <= max_iter):

        if itr == 0:

            for s in range(4):

                current_lambda = lambda_vec[s]

                #Run ssplines with current lambda
                y_tikhonov = tikhonov(y, lambd = current_lambda)
                ress[s], regs[s] = y_tikhonov[1]
    
            normalization_coefs = normalize_get(ress, regs)
            xis, etas = normalize_fit(ress, regs, normalization_coefs)
            
            if normalize:
                lc_res = list(xis)
                lc_reg = list(etas)
            else:
                lc_res = list(ress)
                lc_reg = list(regs)

            P = np.array([[xis[0],xis[1],xis[2],xis[3]], [etas[0],etas[1],etas[2],etas[3]]])              
            indx = np.argsort(lambda_vec)

            #Sort lambdas
            lambda_vec = lambda_vec[indx]
            P = P[:,indx]

        # Compute curvatures of the current points
        C2 = menger(P[:,0], P[:,1], P[:,2])
        C3 = menger(P[:,1], P[:,2], P[:,3])

        # Check if the curvature is negative and update values
        while C3 < 0:

            #Reassign maximum and interior lambdas and Lcurve points (Golden search interval)
            lambda_vec[3] = lambda_vec[2]
            P[:,3] = P[:,2]
            lambda_vec[2] = lambda_vec[1]
            P[:,2] = P[:,1]

            #Update interior lambda and interior point
            lambda_vec[1] = 10 ** ( (np.log10(lambda_vec[3]) + pGS * np.log10(lambda_vec[0])) / (1 + pGS) ) 

            #Run ssplines with current lambda
            y_tikhonov = tikhonov(y, lambd = lambda_vec[1])
            res, reg = y_tikhonov[1]
        
            xi, eta = normalize_fit(res, reg, normalization_coefs)
            
            P[:,1] = [xi,eta]

            C3 = menger(P[:,1], P[:,2], P[:,3])

        # Update values depending on the curvature at the new points
        if C2 > C3:

            current_lambda = lambda_vec[1]

            #Reassign maximum and interior lambdas and Lcurve points (Golden search interval)
            lambda_vec[3] = lambda_vec[2]
            P[:,3] = P[:,2]

            lambda_vec[2] = lambda_vec[1]
            P[:,2] = P[:,1]

            #Update interior lambda and interior point
            lambda_vec[1] = 10 ** ( (np.log10(lambda_vec[3]) + pGS * np.log10(lambda_vec[0])) / (1 + pGS) ) 
            
            y_tikhonov = tikhonov(y, lambd = lambda_vec[1])
            res, reg = y_tikhonov[1] 
        
            xi, eta = normalize_fit(res, reg, normalization_coefs)
        
            P[:,1] = [xi,eta]
            
        else:

            current_lambda = lambda_vec[2]

            #Reassign maximum and interior lambdas and Lcurve points (Golden search interval)
            lambda_vec[0] = lambda_vec[1]
            P[:,0] = P[:,1]

            lambda_vec[1] = lambda_vec[2]
            P[:,1] = P[:,2]

            #Update interior lambda and interior point
            lambda_vec[2] = 10 ** ( np.log10(lambda_vec[0]) + np.log10(lambda_vec[3]) - np.log10(lambda_vec[1]) )
            
            y_tikhonov = tikhonov(y, lambd = lambda_vec[2])
            res, reg = y_tikhonov[1] 

            xi, eta = normalize_fit(res, reg, normalization_coefs)

            P[:,2] = [xi, eta]
            
        gap = ( lambda_vec[3] - lambda_vec[0] ) / lambda_vec[3]

        if gap < epsilon:
            print(f'  Convergence criterion reached in {itr} iterations.')

        lambda_itr.append(current_lambda)
        
        if normalize:
            lc_res.append(xi)
            lc_reg.append(eta)
        else:
            lc_res.append(res)
            lc_reg.append(reg)

        itr += 1

        if itr == max_iter:
            print(f'  Maximum number of {itr} iterations reached.')

    #Solve for optimal lambda
    
    y_tikhonov = tikhonov(y, lambd = current_lambda)
    y_tik = y_tikhonov[0]
    res, reg = y_tikhonov[1] 
    
    if normalize:
        xi, eta = normalize_fit(res, reg, normalization_coefs)
        lc_res.append(xi)
        lc_reg.append(eta)
    else:
        lc_res.append(res)
        lc_reg.append(reg)

    return [y_tik, current_lambda, lambda_itr, (lc_res, lc_reg)]

def full_lcurve(y, lambda_min = 1e-10, lambda_max = 1e10, n_lambdas = 100, normalize = False, plot_lc = False):
    
    residual_lc = np.zeros(n_lambdas)
    reg_residual_lc = np.zeros(n_lambdas)
    m_samples = len(y)
    y_tik_path = np.zeros((m_samples,n_lambdas))

    #Check range of lambdas and compute normalization coefficients
    #lambda_max, lambda_min = check_lambdas(lambda_min, lambda_max, y)
    
    #Generate array of lambdas
    lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), n_lambdas)
        
    if lambdas[0] < lambda_min:
        lambdas[0] = lambda_min 
        
    if lambdas[-1] > lambda_max:
        lambdas[-1] = lambda_max 
        
    #Loop over lambdas
    for i, lambd in enumerate(lambdas):
    
        #Run tikhonov with current lambda
        y_tikhonov = tikhonov(y, lambd = lambd)
        y_tik_path[:,i] = y_tikhonov[0]
        residual_lc[i], reg_residual_lc[i] = y_tikhonov[1]
    
    if normalize:
        
        normalization_coefs = normalize_get(residual_lc, reg_residual_lc)
        xi, eta = normalize_fit(residual_lc, reg_residual_lc, normalization_coefs)
        
        if plot_lc:
            
            plt.plot(xi, eta)
            plt.xlabel(r'$||y - f(x)||_2$')
            plt.ylabel(r'$||D^{2}f(x)||_2$')
        
        return [y_tik, (xi, eta)]
    
    else:
        
        if plot_lc:

            plt.loglog(residual_lc, reg_residual_lc)
            plt.xlabel(r'$||y - f(x)||_2$')
            plt.ylabel(r'$||D^{2}f(x)||_2$')
    
        return [y_tik_path, (residual_lc, reg_residual_lc)]
    
def gcv(y, lambda_min = 1e-10, lambda_max = 1e10, n_lambdas = 100, plot_lc = False):
    
    m_samples = len(y)
    y_tik_path = np.zeros((m_samples,n_lambdas))
    gcv_lambda = np.zeros(n_lambdas)

    #Check range of lambdas and compute normalization coefficients
#     lambda_max, lambda_min = check_lambdas(lambda_min, lambda_max, y, x)
    
    #Generate array of lambdas
    lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), n_lambdas)
        
    if lambdas[0] < lambda_min:
        lambdas[0] = lambda_min 
        
    if lambdas[-1] > lambda_max:
        lambdas[-1] = lambda_max 
        
    #Loop over lambdas
    for i, lambd in enumerate(lambdas):
    
        #Run ssplines with current lambda
        tik = tikhonov(y, lambd = lambd)
        gcv_lambda[i] = tik[2][0]
        y_tik_path[:,i] = tik[0]
        
    #Chose the lambda tat minimizes GCV
    lamb_idx = np.argmin(gcv_lambda)
    lamb_opt = lambdas[lamb_idx]
    
    #Solve for optimal lambda
    tik_opt = tikhonov(y, lambd = lamb_opt)
    y_tik = tik_opt[0]
        
    if plot_lc:

        plt.semilogx(lambdas, gcv_lambda)
        plt.semilogx(lambdas[lamb_idx], gcv_lambda[lamb_idx],'ko')
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$GCV(\lambda)$')
    
    return [y_tik, (lambdas, gcv_lambda), y_tik_path]