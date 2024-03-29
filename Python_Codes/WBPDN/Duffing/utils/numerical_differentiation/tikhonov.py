# IMPORTS
import numpy as np
import cvxpy as cp
from scipy.sparse import coo_matrix, eye, vstack
from scipy.sparse.linalg import spsolve
from scipy.interpolate import PchipInterpolator
from scipy.linalg import norm
import matplotlib.pyplot as plt

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

def check_lambdas(lambda_min, lambda_max, y, x, tol = 1e-8):
    
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
    
    out = tikhonov_diff(y, x, lambd = lambda_min)
    residual_min, reg_residual_max = out[1]
    out = tikhonov_diff(y, x, lambd = lambda_max)
    residual_max, reg_residual_min = out[1]
    
    #Make sure the residuals are not too small to help lcurve_corner converge
    while residual_min < tol:
        print('lambda_min too small. Increasing it 10 times...')
        lambda_min = lambda_min * 10
        out = tikhonov_diff(y, x, lambd = lambda_min)
        residual_min, reg_residual_max = out[1]

    while reg_residual_min < tol:
        print('lambda_max too large. Reducing it 10 times...')
        lambda_max = lambda_max / 10
        out = tikhonov_diff(y, x, lambd = lambda_max)
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

def tikhonov_diff(y, x, lambd = 0, verbose = False):
    
    """
    trend_filter(y, lambda = 0, order = 3)
    
    finds the solution of the l1 trend estimation problem
    
     minimize    (1/2)||y-x||^2+lambda*||Dx||_1,
    
    with variable x, and problem data y and lambda, with lambda >0.
    D is the k-th order difference matrix.
    This function calls cvxpy to solve the optimization problem above
    
    Input arguments:
    
     - y:          n-vector; original signal
     - lambda:     scalar; positive regularization parameter
     - order:      scalar: order of the difference matrices
    
    Output arguments:
    
     - x:          n-vector; primal optimal point
    
    Adapted from
    "l1 Trend Filtering", S. Kim, K. Koh, ,S. Boyd and D. Gorinevsky
    Based on code available at: https://www.cvxpy.org/examples/applications/l1_trend_filter.html
    
    Author: Alexandre Cortiella
    Affiliation: University of Colorado Boulder
    Department: Aerospace Engineering Sciences
    Date: 11/09/2020
    Version: v1.0
    Updated: 11/09/2020
    
    """
    m = len(x)
    n = m-1    
    
    dx = x[1] - x[0]

    #Generate the integral matrix
    row,col = np.tril_indices(n)
    data = dx * np.ones(len(row))
    A = coo_matrix((data, (row, col)), shape=(n,n))

    #Generate the difference operators
    D1 = diff_mat(n,1)/dx
    D2 = diff_mat(n,2)/(dx ** 2)
    D = vstack([eye(n),D1,D2])
    
    #Generate Tikhonov matrix
    T = (A.T * A + lambd * D.T * D)

    #Generate LHS
    yhat = y[1:] - y[0]
    b = A.T * yhat

    #Solve linear system
    u = spsolve(T, b)
    
    #Compute residuals with current solution
    residual = norm(A * u - yhat)
    reg_residual = norm(D * u, ord = 2)
    
    #Interpolate the midpoint derivatives to the nodes using a cubic polynomial
    x_mid = 0.5 * (x[0:-1] + x[1:])
    f = PchipInterpolator(x_mid, u)
    dy = f(x)
    
    return [dy, (residual, reg_residual)]

def lcurve_corner(y, x, lambda_max = 1e10, lambda_min = 1e-10, epsilon = 0.001, max_iter = 50, normalize = False, verbose = False):
    
    n = len(y)
    pGS = (1 + np.sqrt(5))/2 #Golden search parameter
    gap = 1
    itr = 0
    lambda_itr = []

    #Check the range of lambdas and compute normalization coefficients
    lambda_max, lambda_min = check_lambdas(lambda_min, lambda_max, y, x)

    lambda_vec = np.array([lambda_min, lambda_max, 0, 0])

    lambda_vec[2] = 10 ** ( (np.log10(lambda_vec[1]) + pGS * np.log10(lambda_vec[0])) / (1 + pGS) ) 
    lambda_vec[3] = 10 ** ( np.log10(lambda_vec[0]) + np.log10(lambda_vec[1]) - np.log10(lambda_vec[2]) )

    ress = np.zeros(4)#residuals
    regs = np.zeros(4)#regularization residuals

    while (gap >= epsilon) and (itr <= max_iter):

        if itr == 0:

            for s in range(4):

                current_lambda = lambda_vec[s]

                #Run trend filter with current lambda

                out = tikhonov_diff(y, x, lambd = current_lambda)
                ress[s], regs[s] = out[1]
    
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

            out = tikhonov_diff(y, x, lambd = lambda_vec[1])
            res, reg = out[1] 
        
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
            
            out = tikhonov_diff(y, x, lambd = lambda_vec[1])
            res, reg = out[1] 
        
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
            
            out = tikhonov_diff(y, x, lambd = lambda_vec[2])
            res, reg = out[1] 

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
    
    out = tikhonov_diff(y, x, lambd = current_lambda)
    dy = out[0]
    res, reg = out[1]
    
    if normalize:
        xi, eta = normalize_fit(res, reg, normalization_coefs)
        lc_res.append(xi)
        lc_reg.append(eta)
    else:
        lc_res.append(res)
        lc_reg.append(reg)

    return [dy, current_lambda, lambda_itr, (lc_res, lc_reg)]

def full_lcurve(y, x, lambda_max = 1e10, lambda_min = 1e-10, n_lambdas = 100, normalize = False, plot_lc = False):
    
    residual_lc = np.zeros(n_lambdas)
    reg_residual_lc = np.zeros(n_lambdas)
    n = len(x)
    dy = np.zeros((n, n_lambdas))
    
    #Check range of lambdas and compute normalization coefficients
    lambda_max, lambda_min = check_lambdas(lambda_min, lambda_max, y, x, tol = 1e-8)

    #Generate array of lambdas
    lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), n_lambdas)
    
    if lambdas[0] < lambda_min:
        lambdas[0] = lambda_min 
        
    if lambdas[-1] > lambda_max:
        lambdas[-1] = lambda_max 
        
    #Loop over lambdas
    for i, lambd in enumerate(lambdas):
    
        out = tikhonov_diff(y, x, lambd = lambd)
        dy[:,i] = out[0]
        residual_lc[i], reg_residual_lc[i] = out[1]
    
    if normalize:
        
        normalization_coefs = normalize_get(residual_lc, reg_residual_lc)
        xi, eta = normalize_fit(residual_lc, reg_residual_lc, normalization_coefs)
        
        if plot_lc:
            plt.plot(xi, eta)
            plt.xlabel(r'$||Au - \hat{y}||_2$')
            plt.ylabel(r'$||D^{0}u||_1$'.format(order))
        
        return [dy,(xi, eta)]
    
    else:
        
        if plot_lc:
            plt.loglog(residual_lc, reg_residual_lc)
            plt.xlabel(r'$||Au - \hat{y}||_2$')
            plt.ylabel(r'$||D^{0}u||_1$'.format(order))
    
        return [dy, (residual_lc, reg_residual_lc)]