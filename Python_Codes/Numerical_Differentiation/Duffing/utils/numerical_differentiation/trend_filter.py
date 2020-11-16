# IMPORTS
import numpy as np
from scipy.sparse import coo_matrix, eye
import scipy.linalg as sla
from scipy.linalg import norm

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

def trend_filter(y, lambd = 0, order = 3):
    
    """
    trend_filter(y, lambda = 0, order = 3)
    
    finds the solution of the l1 trend estimation problem
    
     minimize    (1/2)||y-x||^2+lambda*||Dx||_1,
    
    with variable x, and problem data y and lambda, with lambda >0.
    D is the second difference matrix, with rows [0... -1 2 -1 ...0]
    
    and the dual problem
    
     minimize    (1/2)||D'*z||^2-y'*D'*z
     subject to  norm(z,inf) <= lambda,
    
    with variable z.
    
    Input arguments:
    
     - y:          n-vector; original signal
     - lambda:     scalar; positive regularization parameter
     - order:      scalar: order of the difference matrices
    
    Output arguments:
    
     - x:          n-vector; primal optimal point
    
    Adaptation from
    see "l1 Trend Filtering", S. Kim, K. Koh, ,S. Boyd and D. Gorinevsky
    www.stanford.edu/~boyd/l1_trend_filtering.html
    
    Author: Alexandre Cortiella
    Affiliation: University of Colorado Boulder
    Department: Aerospace Engineering Sciences
    Date: 11/09/2020
    Version: v1.0
    Updated: 11/09/2020
    
    """
    
    # PARAMETERS
    alpha = 0.01 #backtracking linesearch parameter (0,0.5]
    beta = 0.5 # backtracking linesearch parameter (0,1)
    mu = 2 # IPM parameter: t update
    max_iter = 40 # IPM parameter: max iteration of IPM
    max_ls_iter = 20 # IPM parameter: max iteration of line search
    tol = 1e-4 # IPM parameter: tolerance

    itr = 0
    gap = 1

    # DIMENSIONS
    n   = len(y) #length of signal x

    # OPERATOR MATRICES
    D = diff_mat(n,order)

    DDT = D * D.T
    Dy = D * y

    m = len(Dy)

    # VARIABLES
    z   = np.zeros(m) # dual variable
    mu1 = np.ones(m) # dual of dual variable
    mu2 = np.ones(m) # dual of dual variable

    t = 1e-10; 
    p_obj =  float('inf')
    d_obj =  0
    step =  float('inf')
    f1   =  z - lambd
    f2   = - z - lambd
    print(f'Iteration   Primal obj.    Dual obj.     Gap')
    print('\n')
    
    #----------------------------------------------------------------------
    #               MAIN LOOP
    #----------------------------------------------------------------------

    for iters in range(max_iter):

        DTz = (z.T * D).T
        DDTz = D * DTz

        w = Dy - (mu1 - mu2)

        # two ways to evaluate primal objective:
        # 1) using dual variable of dual problem
        # 2) using optimality condition    
        #temp = lsqr(DDT, w)[0] Not comparable to backslash in matlab (unstable)
        #temp = nla.lstsq(DDT.todense(), w)[0] # numpy library (similar results as scipy)

        temp = sla.lstsq(DDT.todense(), w)[0] #may be an overkill but stable
        p_obj1 = 0.5 * np.dot(w,temp) + lambd * np.sum(mu1 + mu2)
        p_obj2 = 0.5 * np.dot(DTz.T, DTz) + lambd * np.sum(np.abs(Dy - DDTz))

        p_obj = np.min([p_obj1, p_obj2])
        d_obj = -0.5 * np.dot(DTz, DTz) + np.dot(Dy.T, z)

        gap  =  p_obj - d_obj

        print("{0:6d} {1:15.4e} {2:13.5e} {3:10.2e}".format(iters, p_obj, d_obj, gap))
        #Check stopping criterion
        if gap <= tol:
            status = 'solved'
            print(status)
            x = y - D.T * z

            #return x
            break

        if step >= 0.2:
            t = np.max([2 * m * mu / gap, 1.2 * t])

        # CALCULATE NEWTON STEP

        rz = DDTz - w

        val = mu1 / f1 + mu2 / f2
        row = np.arange(m)
        col = np.arange(m)

        S = DDT - coo_matrix((val, (row, col)), shape = (m,m))
        r = - DDTz + Dy + ( 1 / t ) / f1 - ( 1 / t ) / f2

        dz      =  sla.lstsq(S.todense(), r)[0]
        dmu1    = - ( mu1 + ( (1 / t) + dz * mu1 ) / f1 )
        dmu2    = - ( mu2 + ( ( 1 / t ) - dz * mu2 ) / f2 )

        resDual = rz
        resCent = np.concatenate([- mu1 * f1 - 1 / t, - mu2 * f2 - 1 / t])
        residual= np.concatenate([resDual, resCent])

        # BACKTRACKING LINESEARCH
        negIdx1 = (dmu1 < 0)
        negIdx2 = (dmu2 < 0)
        step = 1

        if any(negIdx1):
            step = np.min( [step, 0.99 * np.min( - mu1[negIdx1] / dmu1[negIdx1] )])
        if any(negIdx2):
            step = np.min( [step, 0.99 * np.min( - mu2[negIdx2] / dmu2[negIdx2] )])

        for liter in range(max_ls_iter):

            newz = z  + step * dz
            newmu1 = mu1 + step * dmu1
            newmu2 = mu2 + step * dmu2
            newf1 = newz - lambd
            newf2 = - newz - lambd

            # UPDATE RESIDUAL
            newResDual = DDT * newz - Dy + newmu1 - newmu2
            newResCent = np.concatenate([- newmu1 * newf1 - 1 / t, - newmu2 * newf2 - 1 / t])
            newResidual = np.concatenate([newResDual, newResCent])

            if ( np.max([np.max(newf1), np.max(newf2)]) < 0 ) and ( norm(newResidual) <= (1 - alpha * step) * norm(residual) ):
                break

            step = beta * step

        # UPDATE PRIMAL AND DUAL VARIABLES
        z  = newz
        mu1 = newmu1
        mu2 = newmu2
        f1 = newf1
        f2 = newf2

    # The solution may be close at this point, but does not meet the stopping
    # criterion (in terms of duality gap).
    x = y - D.T * z
    if (iters >= max_iter):
        status = 'maxiter exceeded'
        print(status)
        
    return x