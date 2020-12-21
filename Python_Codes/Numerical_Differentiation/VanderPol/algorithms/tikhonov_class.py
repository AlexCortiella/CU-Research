import numpy as np
from scipy.sparse import coo_matrix, eye, vstack
from scipy.sparse.linalg import spsolve
from scipy.interpolate import PchipInterpolator

class TikhonovDifferentiation():
    
    def __init__(self, alphas = 0):
        
        #Define attributes
        self.alphas = alphas
                
    #METHODS
        
    def fit(self,x,y,alphas=0):
        
        if not isinstance(alphas, list):

            if isinstance(alphas, (int,float)):
                alphas = [alphas]
            elif isinstance(alphas, (tuple, np.ndarray)):
                alphas = list(alphas)
            else:
                print('Unsupported data type.')

        #Define attributes
        self.alphas = alphas
    
        #Local function to compute the difference matrices of anu order
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
        
        #Tikhonov differentiation
        m = len(x)
        n = m-1
        
        self.alphas = alphas
        n_alphas = len(alphas)
        dy_alpha = np.zeros((len(x),n_alphas))
        residual = np.zeros(n_alphas)
        reg_residual = np.zeros(n_alphas)
        diagS = np.zeros((n,n_alphas))
        gcv = np.zeros(n_alphas)
        
        self.x_mid = 0.5 * (x[0:-1] + x[1:])
        self.u_path = np.zeros((n,n_alphas))

        dx = x[1] - x[0]

        #Generate the integral matrix
        row,col = np.tril_indices(n)
        data = dx * np.ones(len(row))
        A = coo_matrix((data, (row, col)), shape=(n,n))

        #Generate the difference operators
        D1 = diff_mat(n,1)/dx
        D2 = diff_mat(n,2)/(dx ** 2)
        D = vstack([eye(n),D1,D2])
        
        for i, alpha in enumerate(alphas): 
            #Generate Tikhonov matrix
            T = (A.T * A + alpha * D.T * D)

            #Generate LHS
            yhat = y[1:] - y[0]
            b = A.T * yhat

            #Solve linear system
            u = spsolve(T, b)
            self.u_path[:,i] = u
            
            #Compute residuals
            residual[i] = np.linalg.norm(A * u - yhat)
            reg_residual[i] = np.linalg.norm(D * u)
            
            #Compute GCV score
            S = A * T * A.T
            gcv[i] = 1/n * np.sum(((yhat - A * u)/(1 - S.diagonal().sum()/n)) ** 2)
            diagS[:,i] = S.diagonal()#Compute diagonal elements of the projector fot CV and GCV
            
            #Interpolate the midpoint derivatives to the nodes using a cubic polynomial
            f = PchipInterpolator(self.x_mid, u)
            dy = f(x)
            
            dy_alpha[:,i] = dy
            
        #Store residuals as attributes    
        self.residual = residual
        self.reg_residual = reg_residual
        self.diagS = diagS
        self.gcv = gcv
        
        return dy_alpha
    
    def predict(self,x_new):
        
        n_alphas = len(self.alphas)
        dy_pred = np.zeros((len(self.x),n_alphas))
        
        for i in range(n_alphas): 
            #Interpolate the midpoint derivatives to the new nodes using a cubic polynomial
            f = PchipInterpolator(self.x_mid, self.u_path[:,i])
            dy_pred[:,i] = f(x_new)
        
        return dy_pred
    
        