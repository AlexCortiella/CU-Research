# IMPORTS
import numpy as np
from scipy.sparse import coo_matrix, eye
import scipy.linalg as sla
from scipy.linalg import norm
import matplotlib.pyplot as plt
from scipy import interpolate

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr

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

# def check_lambdas(lambda_min, lambda_max, y, x, tol = 1e-8):
       
#     # Lambda_min
#     ss_min = ssplines(y, x, lambd = lambda_min)
#     residual_min, reg_residual_max = ss_min[1]
    
#     # Lambda_max
#     ss_max = ssplines(y, x, lambd = lambda_max)
#     residual_max, reg_residual_min = ss_max[1]
    
#     while residual_min < tol:
#         lambda_min = lambda_min * 10
#         ss_min = ssplines(y, x, lambd = lambda_min)
#         residual_min, reg_residual_max = ss_min[1]

#     while reg_residual_min < tol:
#         lambda_max = lambda_max / 10
#         ss_max = ssplines(y, x, lambd = lambda_max)
#         residual_max, reg_residual_min = ss_max[1]
    
#     #Compute normalization coefficients to transform the L-curve into a [-1, 1] range
#     cres0 = 1 - 2/(np.log(residual_max) - np.log(residual_min)) * np.log(residual_max)
#     cres1 = 2 / (np.log(residual_max) - np.log(residual_min))
#     creg0 = 1 - 2/(np.log(reg_residual_max) - np.log(reg_residual_min)) * np.log(reg_residual_max)
#     creg1 = 2 / (np.log(reg_residual_max) - np.log(reg_residual_min))
    
#     normalization_coefs = [(cres0, cres1),(creg0, creg1)]
        
#     return (lambda_max, lambda_min, normalization_coefs)

def check_lambdas(lambda_min, lambda_max, y, x, tol = 1e-8):
    
    #Make sure that the solver does not fail for the range of lambdas 
    while True:
        try:
            ss_min = ssplines(y, x, lambd = lambda_min)
            residual_min, reg_residual_max = ss_min[1]
        except:
            print('lambda_min too small. Increasing it 10 times...')
            lambda_min = lambda_min * 10
            continue
        else:
            break
    
    while True:
        try:
            ss_max = ssplines(y, x, lambd = lambda_max)
            residual_max, reg_residual_min = ss_max[1]
        except:
            print('lambda_max too large. Reducing it 10 times...')
            lambda_max = lambda_max / 10
            continue
        else:
            break
    
    #Make sure the residuals are not too small to help lcurve_corner converge
    while residual_min < tol:
        print('lambda_min too small. Increasing it 10 times...')
        lambda_min = lambda_min * 10
        ss_min = ssplines(y, x, lambd = lambda_min)
        residual_min, reg_residual_max = ss_min[1]

    while reg_residual_min < tol:
        print('lambda_max too large. Reducing it 10 times...')
        lambda_max = lambda_max / 10
        ss_max = ssplines(y, x, lambd = lambda_max)
        residual_max, reg_residual_min = ss_max[1]
            
    return (lambda_max, lambda_min)

# def normalize_lcurve(residual, reg_residual, normalization_coefs):
    
#     #Unpack coefficients
#     cres0, cres1 = normalization_coefs[0]
#     creg0, creg1 = normalization_coefs[1]
    
#     xi = cres0 + cres1 * np.log(residual)
#     eta = creg0 + creg1 * np.log(reg_residual)
    
#     return (xi, eta)

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

def ssplines(y, x, lambd = 0):
    
    """
    trend_filter(y, lambda = 0, order = 3)
    
    finds the solution of the l1 trend estimation problem
    
     minimize    (1/2)||y-f(x)||^2 + lambda*||D^2 f(x)||_2^2,
    
    with variable x, and problem data y and lambda, with lambda >0.
    This function uses rpy2 to call the smoothing splines implementation in R
    
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
    # Transform into R vectorsfrom scipy import interpolate
    r_y = robjects.FloatVector(y)
    r_x = robjects.FloatVector(x)
    
    #Create ssplines object with specific inputs
    r_smooth_spline = robjects.r['smooth.spline'] #extract R function
    kwargs = {"x": r_x, "y": r_y, "lambda": float(lambd)}
    spline1 = r_smooth_spline(**kwargs)
        
    #Compute filtered signal for a specific lambda
    y_ss = np.array(spline1.rx2('y'))
    
    #Compute its derivative
    y_ss_ss = interpolate.splrep(r_x, y_ss, k=3, s=0)
    dy_ss = interpolate.splev(r_x, y_ss_ss, der=1)
    ddy_ss = interpolate.splev(r_x, y_ss_ss, der=2)
    
    #Compute residuals
    residual = norm(y - y_ss)
    reg_residual = norm(ddy_ss)
   
    return [(y_ss, dy_ss), (residual, reg_residual)]

# def lcurve_corner(y, x, order = 3, lambda_max = 1000, lambda_min = 1e-10, epsilon = 0.001, max_iter = 50, normalize = True, verbose = False):
    
#     n = len(y)
#     pGS = (1 + np.sqrt(5))/2 #Golden search parameter
#     gap = 1
#     itr = 0
#     lambda_itr = []

#     #Check the range of lambdas and compute normalization coefficients
#     lambda_max, lambda_min, normalization_coefs = check_lambdas(lambda_min, lambda_max, y, x)

#     lambda_vec = np.array([lambda_min, lambda_max, 0, 0])
#     lambda_vec[2] = np.exp( (np.log(lambda_vec[1]) + pGS * np.log(lambda_vec[0])) / (1 + pGS) ) 
#     lambda_vec[3] = np.exp( np.log(lambda_vec[0]) + np.log(lambda_vec[1]) - np.log(lambda_vec[2]) )

#     residuals = np.zeros(4)
#     reg_residuals = np.zeros(4)
#     lc_res = []
#     lc_reg = []
    
#     D = diff_mat(n,order)


#     while (gap >= epsilon) and (itr <= max_iter):

#         if itr == 0:

#             for s in range(4):

#                 current_lambda = lambda_vec[s]

#                 #Run ssplines with current lambda
#                 sspline = ssplines(y, x, lambd = current_lambda)
#                 residuals[s], reg_residuals[s] = sspline[1]

#             #Normalize between -1 and 1
#             xis, etas = normalize_lcurve(residuals, reg_residuals, normalization_coefs)
            
#             if normalize:
#                 lc_res.append(xis)
#                 lc_reg.append(etas)
#             else:
#                 lc_res.append(residuals)
#                 lc_reg.append(reg_residuals)

#             P = np.array([[xis[0],xis[1],xis[2],xis[3]], [etas[0],etas[1],etas[2],etas[3]]])           
#             indx = np.argsort(lambda_vec)

#             #Sort lambdas
#             lambda_vec = lambda_vec[indx]
#             P = P[:,indx]

#         # Compute curvatures of the current points
#         C2 = menger(P[:,0], P[:,1], P[:,2])
#         C3 = menger(P[:,1], P[:,2], P[:,3])

#         # Check if the curvature is negative and update values
#         while C3 < 0:

#             #Reassign maximum and interior lambdas and Lcurve points (Golden search interval)
#             lambda_vec[3] = lambda_vec[2]
#             P[:,3] = P[:,2]
#             lambda_vec[2] = lambda_vec[1]
#             P[:,2] = P[:,1]

#             #Update interior lambda and interior point
#             lambda_vec[1] = np.exp( (np.log(lambda_vec[3]) + pGS * np.log(lambda_vec[0])) / (1 + pGS) ) 

#             #Run ssplines with current lambda
#             sspline = ssplines(y, x, lambd = lambda_vec[1])
#             residual, reg_residual = sspline[1]

#             #Normalize between -1 and 1
#             xi, eta = normalize_lcurve(residual, reg_residual, normalization_coefs)

#             P[:,1] = [xi,eta]

#             C3 = menger(P[:,1], P[:,2], P[:,3])

#         # Update values depending on the curvature at the new points
#         if C2 > C3:

#             current_lambda = lambda_vec[1]

#             #Reassign maximum and interior lambdas and Lcurve points (Golden search interval)
#             lambda_vec[3] = lambda_vec[2]
#             P[:,3] = P[:,2]

#             lambda_vec[2] = lambda_vec[1]
#             P[:,2] = P[:,1]

#             #Update interior lambda and interior point
#             lambda_vec[1] = np.exp( (np.log(lambda_vec[3]) + pGS * np.log(lambda_vec[0])) / (1 + pGS) ) 

#             #Run ssplines with current lambda
#             sspline = ssplines(y, x, lambd = lambda_vec[1])
#             residual, reg_residual = sspline[1]

#             #Normalize between -1 and 1
#             xi, eta = normalize_lcurve(residual, reg_residual, normalization_coefs)

#             P[:,1] = [xi,eta]

#         else:

#             current_lambda = lambda_vec[2]

#             #Reassign maximum and interior lambdas and Lcurve points (Golden search interval)
#             lambda_vec[0] = lambda_vec[1]
#             P[:,0] = P[:,1]

#             lambda_vec[1] = lambda_vec[2]
#             P[:,1] = P[:,2]

#             #Update interior lambda and interior point
#             lambda_vec[2] = np.exp( np.log(lambda_vec[0]) + np.log(lambda_vec[3]) - np.log(lambda_vec[1]) )

#             #Run ssplines with current lambda
#             sspline = ssplines(y, x, lambd = lambda_vec[2])
#             residual, reg_residual = sspline[1]

#             #Normalize between -1 and 1
#             xi, eta = normalize_lcurve(residual, reg_residual, normalization_coefs)

#             P[:,2] = [xi,eta]

#         gap = ( lambda_vec[3] - lambda_vec[0] ) / lambda_vec[3]

#         if gap < epsilon:
#             print(f'  Convergence criterion reached in {itr} iterations.')

#         lambda_itr.append(current_lambda)

#         if normalize:
#             lc_res.append(xi)
#             lc_reg.append(eta)
#         else:
#             lc_res.append(residual)
#             lc_reg.append(reg_residual)

#         itr += 1

#         if itr == max_iter:
#             print(f'  Maximum number of {itr} iterations reached.')

#     #Solve for optimal lambda
#     #Run ssplines with current lambda
#     sspline = ssplines(y, x, lambd = current_lambda)
#     y_ss, dy_ss = sspline[0]
    
#     return [y_ss, dy_ss, current_lambda, lambda_itr, (lc_res, lc_reg)]

def lcurve_corner(y, x, lambda_min = 1e-10, lambda_max = 1e10, epsilon = 0.001, max_iter = 50, normalize = False, verbose = False):
    
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

                #Run ssplines with current lambda
                sspline = ssplines(y, x, lambd = current_lambda)
                ress[s], regs[s] = sspline[1]
    
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
            sspline = ssplines(y, x, lambd = lambda_vec[1])
            res, reg = sspline[1]
        
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
            
            sspline = ssplines(y, x, lambd = lambda_vec[1])
            res, reg = sspline[1] 
        
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
            
            sspline = ssplines(y, x, lambd = lambda_vec[2])
            res, reg = sspline[1] 

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
    
    sspline = ssplines(y, x, lambd = current_lambda)
    y_ss, dy_ss = sspline[0]
    res, reg = sspline[1] 
    
    if normalize:
        xi, eta = normalize_fit(res, reg, normalization_coefs)
        lc_res.append(xi)
        lc_reg.append(eta)
    else:
        lc_res.append(res)
        lc_reg.append(reg)

    return [y_ss, dy_ss, current_lambda, lambda_itr, (lc_res, lc_reg)]

def full_lcurve(y, x, lambda_min = 1e-10, lambda_max = 1e10, n_lambdas = 100, normalize = False, plot_lc = False):
    
    residual_lc = np.zeros(n_lambdas)
    reg_residual_lc = np.zeros(n_lambdas)
    m_samples = len(y)
    y_ss = np.zeros((m_samples,n_lambdas))
    dy_ss = np.zeros((m_samples,n_lambdas))

    #Check range of lambdas and compute normalization coefficients
    lambda_max, lambda_min = check_lambdas(lambda_min, lambda_max, y, x)
    
    #Generate array of lambdas
    lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), n_lambdas)
        
    if lambdas[0] < lambda_min:
        lambdas[0] = lambda_min 
        
    if lambdas[-1] > lambda_max:
        lambdas[-1] = lambda_max 
        
    #Loop over lambdas
    for i, lambd in enumerate(lambdas):
    
        #Run ssplines with current lambda
        sspline = ssplines(y, x, lambd = lambd)
        y_ss[:,i], dy_ss[:,i] = sspline[0]
        residual_lc[i], reg_residual_lc[i] = sspline[1]
    
    if normalize:
        
        normalization_coefs = normalize_get(residual_lc, reg_residual_lc)
        xi, eta = normalize_fit(residual_lc, reg_residual_lc, normalization_coefs)
        
        if plot_lc:
            
            plt.plot(xi, eta)
            plt.xlabel(r'$||y - f(x)||_2$')
            plt.ylabel(r'$||D^{2}f(x)||_2$')
        
        return (xi, eta)
    
    else:
        
        if plot_lc:

            plt.loglog(residual_lc, reg_residual_lc)
            plt.xlabel(r'$||y - f(x)||_2$')
            plt.ylabel(r'$||D^{2}f(x)||_2$')
    
        return [y_ss, dy_ss, (residual_lc, reg_residual_lc)]