"""
This generates Gibbs samples based on Potts model

Each sample is a row vector of size dim

The elements can take the value of {0, 1, ..., (q - 1)}
""" 


"""
3 steps to resample each row vector elements(q = 3):

    1- Draw a random -- U in [0, 1]
    
    2- Build the cumulative probability array -- {P_0, P_0 + P_1, 1}
       
       P_0 means the probability of getting 0 for the element 
       resampled conditioned on the rest of elements in the vector
       
    3- Assign the new data point -- 0 if U < P_0
                                    1 if P_0 < U < P_0 + P_1
                                    2 if P_0 + P_1 < U < 1                            
"""                


from MPF_Fitter import MPF


import numpy as np
    
    

class sampler(MPF):
    
    def __init__(self, q, dim, rng):
        self.q = q
        self.dim = dim
        self.rng = rng
        
        
        
    def cumul_prob(self, j, row, J):
    
        """
        This function calculate the probability of
      
        choosing 0, 1, ..., (q - 1) for the element 
    
        jth of the row vector conditioned on the rest 
    
        of elements with respect to the a coupling 
    
        matrix. ased on Potts energy function:
    
        P ~ exp[-J_ij(delta(x_i, x_j))]
        """
    
    
        # Initialize the array with zeros minus the bias
    
        # Each element in E shows the probability of choosing: (its index + 1) 
    
        E = np.zeros(self.q) - J[j].sum()

    
        # Scan the entire row and for each element add the coupling to the corresponding probability
    
        # e. g. if the elemnt ith is 2 add J_orig[j, 2] to E[2] (times 2 for bias)
    
        for i in xrange(len(row)): 
            E[row[i]] += 2 * J[j, i]
     
    
        # Take the exp of the negative energy

        P_Potts = np.exp(-E)
    
        # The function returns the cumulative probabilities normalized to 1
    
        P_Potts = np.cumsum(P_Potts) 
        return P_Potts / P_Potts[self.q - 1]


    
        
    def sampler(self, J, nsamples):
        
        n_burn_in = self.dim * 100000
        sample_space = self.dim * 10
        total_samples = n_burn_in + sample_space * (nsamples - 1)
        
        
        # Initialize the first sample vector by multinomial random choice

        
        choice = [i for i in xrange(self.q)]
        row = self.rng.choice(choice, size = self.dim, p = [1./self.q]*self.q)


        
        samples = np.zeros((nsamples, self.dim))
        sample_idx = 0

        
        for i in xrange(total_samples):
            
            # Pick the element to resample randomly
            
            j = self.rng.randint(self.dim)
            
            
            # Resample that, which is:
            
            # 1- Draw a random 
            
            U = self.rng.uniform()

            
            
            # 2- Build the cumulative probability array
            
            P = self.cumul_prob(j, row, J)

    
            # 3- Assign the new data point
    
            count = 0
            while U > P[count]: count += 1

            row[j] = count
            row_new = row
            
            
            # After large enough number of iteration accept the row and start sampling the next row
  
            if (i >= n_burn_in) and ((i - n_burn_in) % sample_space == 0):
                samples[sample_idx] = row_new
                sample_idx += 1
         
        
        
        return samples

