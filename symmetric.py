import numpy as np
from scipy.optimize import minimize

from math import sqrt, factorial
from time import time
from basics import Multinomial, kPartitions

class ProductBasis():
    ''' Represents a basis of the symmetric subspace of n qudits using product states.
        After  ConstructBasis was executed,
            - self.raw_vectors contains k d-dimensional vectors such that these vectors to the tensor-power of n form a basis of the symmetric subspace. We call this the "raw" basis.
            - self.basis_coeff gives coefficients of an ONB in the raw basis.
        After ConstructOccBasis,
            - self.occ_coeff contains coefficients of the occupation number basis in the raw basis. '''

    def __init__(self, n, d, dtype = np.complex128, seed = None, optimizer_max_calls = 100, max_orthonormality_error = 10 ** -12, verbose = False):
        self.n, self.d, self.dtype, self.maxfun = n, d, dtype, optimizer_max_calls
        startt = time()        

        self.k = Multinomial((n, d - 1)) # Dimension of symmetric subspace
        self.RandomState = np.random.RandomState(seed) # Seed RandomState to be used by this class

        self.ConstructBasis()
        error = self.OrthonormalityError()
        assert(error < max_orthonormality_error)

        self.ConstructOccBasis()

        if verbose: print('ProductBasis with n = {0}, d = {1} and error {2} created in {3} seconds.'.format(n, d, error, round(time() - startt, 2)))

    def __repr__(self):
        return '<ProductBasis with n = {0}, d = {1}>'.format(self.n, self.d)

    def ConstructBasis(self):
        n, d, k = self.n, self.d, self.k
       
        ## Initialize everything
        # Raw vectors: First the d obvious vectors and the k-d remaining empty, will be determined in the second part
        self.raw_vectors = np.zeros((k, d), dtype = self.dtype)
        for i in range(d):
            self.raw_vectors[i, i] = 1.0

        # Coefficients of the product basis in the raw basis
        self.basis_coeff = np.zeros((k, k), dtype = self.dtype)
        for i in range(d):
            self.basis_coeff[i, i] = 1.0

        ## Find suitable raw vectors and basis coefficients
        for i in range(d, k):
            # Define the optimizer (function has the side of effect of saving the new raw vector and the new basis coefficients)
            def optimize_target(real_coeff):
                self.raw_vectors[i] = real_coeff.reshape(d, 2).view(self.dtype)[..., 0] # Initialize new vector using d * 2 real numbers
                self.raw_vectors[i] /= np.linalg.norm(self.raw_vectors[i]) # Normalize new vector

                self.raw_gramian = np.dot(self.raw_vectors.conj(), self.raw_vectors.T) ** self.n # Calculate scalar products of the vectors ** n

                # Start Gram-Schmidt procedure (stable version)
                self.basis_coeff[i][i] = 1.0
                for j in range(i):
                    sp = self.ScalarProduct(j, i)
                    self.basis_coeff[i] -= sp * self.basis_coeff[j]

                # Normalize and return negative of norm (because we want to MAXIMIZE the remaining norm)
                norm = sqrt(np.real(self.ScalarProduct(i, i)))
                self.basis_coeff[i] /= norm
                return -norm

            # Call the optimizer
            start_real_coeff = self.RandomState.rand(d * 2)
            minimize(optimize_target, start_real_coeff , method = 'L-BFGS-B', bounds = [(-1.0, 1.0)] * d * 2, options = {'maxfun':self.maxfun})

    def ConstructOccBasis(self):
        
        gramian = np.ones((self.k, self.k), dtype = self.dtype)

        for occ_i, occ in enumerate(kPartitions(self.n, self.d)):
            gramian[occ_i] *= factorial(self.n) / np.prod([factorial(x) for x in occ]) # Number of summands of the occupation number state
            for raw_i in range(self.k):
                for i in range(self.d):
                    gramian[occ_i, raw_i] *= np.conjugate(self.raw_vectors[raw_i, i]) ** occ[i]

        self.occ_coeff = np.linalg.multi_dot([gramian, self.basis_coeff.conj().T, self.basis_coeff])

    def ScalarProduct(self, i, j):
        ''' Compute scalar product of ONB vectors i and j using current coefficients and precompuated values for the raw scalar product '''
        sp = np.dot(np.conjugate(self.basis_coeff[i]), np.dot(self.raw_gramian, self.basis_coeff[j])) # This is slightly faster than einsum, multi_dot or different order of operations
        return sp

    def OrthonormalityError(self):
        ''' Compute matrix of all scalar products and calculate difference with unit matrix '''
        gramian = np.linalg.multi_dot([np.conjugate(self.basis_coeff), self.raw_gramian, self.basis_coeff.T])
        error = np.linalg.norm(gramian - np.eye(self.k))
        return error

if __name__ == '__main__':
    startt = time()
    
    PB = ProductBasis(6, 4, verbose = True)
