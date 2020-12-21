# IMPORTS
import numpy as np
from scipy.sparse import coo_matrix, eye
import scipy.linalg as sla
import matplotlib.pyplot as plt 

from scipy.linalg import norm
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

def find_nearest(array, value):
    "Element in nd array closest to the scalar value"
    idx = np.abs(array - value).argmin()
    return (array[idx], idx)

def normalize_lcurve(residual, reg_residual):
    
    min_res = np.min(residual_lc)
    max_res = np.max(residual_lc)
    min_reg = np.min(reg_residual_lc)
    max_reg = np.max(reg_residual_lc)
    
    cres0 = 1 - 2/(np.log(max_res) - np.log(min_res)) * np.log(max_res)
    cres1 = 2 / (np.log(max_res) - np.log(min_res))
    creg0 = 1 - 2/(np.log(max_reg) - np.log(min_reg)) * np.log(max_reg)
    creg1 = 2 / (np.log(max_reg) - np.log(min_reg))
    
    xi = cres0 + cres1 * np.log(residual)
    eta = creg0 + creg1 * np.log(reg_residual)
    
    return (xi, eta)

def trend_filter(y, x, lambd = 0, order = 3):
    
    """
    trend_filter(y, lambda = 0, order = 3)
    
    finds the solution of the l1 trend estimation problem
    
     minimize    (1/2)||y-x||^2+lambda*||Dx||_1,
    
    with variable x, and problem data y and lambda, with lambda >0.
    This function uses rpy2 to call the trend filter implementation in R
    
    Input arguments:
    
     - y:          n-vector; original signal, dependent variable y(x)
     - x:          n-vector; independent variable
     - lambda:     scalar; positive regularization parameter
     - order:      scalar: order of the difference matrices
    
    Output arguments:
    
     list[0]
     - y_tf:          n-vector; filtered solution
     - dy_tf:         n-vector; filtered derivative
     
     list[1]
     - residual:      l-2 norm of (y - y_tf)
     - reg_residual:  l-1 norm of D * y_tf
    
    Author: Alexandre Cortiella
    Affiliation: University of Colorado Boulder
    Department: Aerospace Engineering Sciences
    Date: 11/09/2020
    Version: v1.0
    Updated: 11/09/2020
    
    """
    matrix = importr('Matrix')
    genlasso = importr('genlasso')
    
    # Transform into R vectorsfrom scipy import interpolate
    r_y = robjects.FloatVector(y)
    r_t = robjects.FloatVector(x)

    #Create trendfilter object with specific inputs
    trendfilter = genlasso.trendfilter(y = r_y, ord = order, maxsteps = 10000, minlam = 0)
    
    #Evaluate fitted data to specific grid r1_t ( Need to convert into np.array )
    lamb_lasso_path = np.array(trendfilter.rx2('lambda'))
    #Find nearest input lambda to array given by the lasso path
    lambd, idx_lambd = find_nearest(lamb_lasso_path, lambd)
    
    #Compute filtered signal for a specific lambda
    y_tf = np.array(trendfilter.rx2('fit'))[:,idx_lambd]
    
    #Compute its derivative
    y_tf_ss = interpolate.splrep(x, y_tf, k=3, s=0)
    dy_tf = interpolate.splev(x, y_tf_ss, der=1)
    
    #Compute residuals
    n = len(y)
    D = diff_mat(n,order)
    
    residual = norm(y - y_tf)
    reg_residual = norm(D * y_tf, ord = 1)
   
    return [(y_tf, dy_tf), (residual, reg_residual)]

def full_lcurve(y, x, order = 3, normalize = False):
    
    matrix = importr('Matrix')
    genlasso = importr('genlasso')
    
    # Transform into R vectorsfrom scipy import interpolate
    r_y = robjects.FloatVector(y)
    r_t = robjects.FloatVector(x)

    #Create trendfilter object with specific inputs
    trendfilter = genlasso.trendfilter(y = r_y, ord = order, maxsteps = 10000, minlam = 0)
    y_tf = np.array(trendfilter.rx2('fit'))
    
    #Evaluate fitted data to specific grid r1_t ( Need to convert into np.array )
    lamb_lasso_path = np.array(trendfilter.rx2('lambda'))
    
    n_lambdas = len(lamb_lasso_path)
      
    residual_lc = np.zeros(n_lambdas)
    reg_residual_lc = np.zeros(n_lambdas)
    
    n = len(y)
    D = diff_mat(n,order)
    
    for i in range(n_lambdas):
        residual_lc[i] = norm(r_y - y_tf[:,i])
        reg_residual_lc[i] = norm(D * y_tf[:,i], ord = 1)
            
    if normalize:
        
        xi, eta = normalize_lcurve(residual_lc, reg_residual_lc)
        
        plt.plot(xi, eta)
        plt.xlabel(r'$||y - \beta||_2$')
        plt.ylabel(r'$||D^{}\beta||_1$'.format(order))
        
        return (xi, eta)
    
    else:
        
        plt.loglog(residual_lc, reg_residual_lc)
        plt.xlabel(r'$||y - \beta||_2$')
        plt.ylabel(r'$||D^{}\beta||_1$'.format(order))
    
        return (residual_lc, reg_residual_lc)
    
def gcv(y, x, order = 3, min_lambda = 0):
    
    matrix = importr('Matrix')
    genlasso = importr('genlasso')
    
    # Transform into R vectorsfrom scipy import interpolate
    r_y = robjects.FloatVector(y)
    r_t = robjects.FloatVector(x)

    #Create trendfilter object with specific inputs
    trendfilter = genlasso.trendfilter(y = r_y, ord = order, maxsteps = 10000, minlam = min_lambda)
    y_tf = np.array(trendfilter.rx2('fit'))
    
    #Evaluate fitted data to specific grid r1_t ( Need to convert into np.array )
    df = np.array(trendfilter.rx2('df'))
    lambdas = np.array(trendfilter.rx2('lambda'))
    n_lambdas = len(df)
    GCV = np.zeros(n_lambdas)
    
    m_samples = len(y)
    
    for i in range(n_lambdas):
        GCV[i] = 1/m_samples * np.sum(((r_y - y_tf[:,i])/(1 - df[i]/m_samples)) ** 2)
        
    min_idx = np.argmin(GCV)
        
    plt.loglog(lambdas, GCV)
    plt.loglog(lambdas[min_idx], GCV[min_idx],'ro')
    plt.ylabel(r'$GCV(\lambda)$')
    plt.xlabel(r'$\lambda$')
    
    return [y_tf[:,min_idx], (lambdas, GCV), y_tf]