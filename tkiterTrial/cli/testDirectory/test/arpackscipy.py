import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from scipy.sparse import random
from scipy import stats
from scipy.stats import rv_continous

class CustomRandomState(np.random.RandomState):
    def randint(self, k):
        i = np.random.randint(k)
        j = np.random.randint(k)
        print i
        return i - i %2

class complexRandom(rv_continous):
    def _pdf(self,x):
        return np.complex(1.0,1.0)
    
np.set_printoptions(suppress=False)

np.random.seed(0)
#X = np.random.random((10000,10000)) - 0.5
rs = CustomRandomState()
rvs = stats.poisson(25, loc=10).rvs
print ("populate")
S = random(1000, 1000, density=0.15, random_state=rs, data_rvs=rvs)
print ("multiply")

X = np.dot(S, S.T) #create a symmetric matrix
print ("eigvec")
evals_large, evecs_large = eigsh(X, 3, which='LM')
print(evals_large)
