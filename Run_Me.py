"""
==================
MPF on Potts Model
==================

There are 4 steps:

   1- Generate the Original coupling matrix randomly

   2- Generate a sample from the above matrix

   3- Fit MPF to the sample and find the Fit coupling matrix

   4- Compare the Original and Fit coupling matrices

"""


from Gibbs_Potts import sampler
from MPF_Fitter import MPF
from Plots import coupling, covariance

import numpy as np




def main():
    """
    Parameters
    ----------


    q       : Potts model parameter -- q equals 2 converges to Ising model 

    dim     : size of the coupling matrices 

    nsample : sample size -- for bigger q should be bigger

    seed    : for consistancy

    """

    q = 10
    
    dim = 40
    
    nsamples = 50000
    
    seed = 13
    
    
    
    # 1- Generate the original coupling matrix randomly
    
    rng = np.random.RandomState(seed)
    
    J_orig = rng.randn(dim, dim)/np.sqrt(dim)
    np.fill_diagonal(J_orig, 0.)

    J_orig = (J_orig + J_orig.T) / 2.
    

    
    # 2- Generate a sample from the above matrix 
    
    o = sampler(q, dim, rng)
    samples_orig = o.sampler(J_orig, nsamples)

    
    
    # 3- Fit MPF to the sample and find the Fit coupling matrix
    
    J_fit = o.minimizer(samples_orig)
    
    
    # 4- Compare the Original and Fit coupling matrices

    coupling(J_orig, J_fit)
    
    
    # Also compare the covariance of the samples from Original and Fit coupling matrices
    
    samples_fit = o.sampler(J_fit, nsamples)
    covariance(J_orig, J_fit, samples_orig, samples_fit)
    
    print (J_orig - J_fit).min(), (J_orig - J_fit).max()
    
    
    
    
if __name__ == '__main__':
    
    main()    
    
