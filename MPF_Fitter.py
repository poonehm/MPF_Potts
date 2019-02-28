"""
This returns the coupling matrix of an input sample,

by first calculating the objective function and then

minimizing it to get the parameters.
"""


import theano
import theano.tensor as T
import numpy as np 
from scipy import optimize




class MPF(object):
    
    
    def __init__(self, q, dim, rng):
        self.q = q
        self.dim = dim
        self.rng = rng
    
    def objective(self, nsamples):
        """
        This returns the MPF_Potts objective function:
        
        exp(sum_j[J_ij(delta(x_i, x_j) - delta(x_i, x_j + 1)) - J_jj])
        
        derivation available at MPF_Potts_Obj.pdf
        """
        
        
        
        # Consider J as a vector rather than matrix due to theano limit
        
        J = T.dvector('J')
        X = T.dmatrix('X')
        
        
        # This returns only diagonal elements of J in form of a vector 

        J_rav = T.reshape(J, (self.dim, self.dim))
        b = T.identity_like(J_rav)
        diag = T.reshape(b, (1, self.dim**2))
        
        
        
        # For one sample vector the delta function of each element with every other element is needed
        
        # Calculate elementwise delta of tile and repeat--tile in opposite direction-- of each vector
        
        # This keeps track of every two pair possible from the sample

        Tile = T.tile(X, (1, self.dim))
        Rep = T.repeat(X, self.dim, axis=1)
        
        
        
        # One bit up which is used in the second term of objective function
        
        # This changes q to zero since q - 1 is the biggest allowed

        Rep2 = Rep + 1
        idxs = (T.eq(Rep2, self.q)).nonzero()
        Rep2 = T.set_subtensor(Rep2[idxs], 0)


        
        # The elementwise delta function 

        Kfull = (T.switch(T.eq(Tile, Rep), 1 , 0) - T.switch(T.eq(Tile, Rep2), 1 , 0)) * J - diag * J


        # Finally the objective function and its gradiant
        
        K = T.exp(T.reshape(Kfull, (self.dim*nsamples, self.dim)).sum(axis = 1)).mean() 
        dK = T.grad(K, J)


        func = theano.function([J, X], [K, dK])
        
        
        return func

        
    def minimizer(self, samples): 
        """
        This builds the objective function from the input
        
        then calculate its minimum argument using BFGS
        
        in this case it is the Fit coupling matrix.
        """
        
        init_vec = np.zeros(self.dim**2)
        func = self.objective(len(samples))
        
        
        res = optimize.minimize(func, init_vec, args=(samples,), jac=True,options={'maxiter':1e+5,'disp':True})
 

        J_fit = res.x.reshape(self.dim, self.dim)
        np.fill_diagonal(J_fit, 0)
        J_fit = (J_fit + J_fit.T)/2
        
        
        return J_fit
        
