#!/usr/bin/python

import numpy as np
from scipy.sparse import dok_matrix

from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
import time
import sys


fileName = "matrix.dat"
if len(sys.argv) == 2:
    fileName = sys.argv[1]
dimension = 0
matFile = open(fileName,"r")

for line in matFile:
    if line.startswith('#'):
        continue
    if dimension ==0:
        dimension = int(line)
        S = dok_matrix((dimension, dimension), dtype=np.complex)
        print S.get_shape()
        a = np.count_nonzero(S.toarray())
        continue
    numbers = line.split()
    row = int(numbers[0])
    col = int(numbers[1])
    vals = numbers[2].split(',')
    real = vals[0]
    real = float(real[1:])
    imag = vals[1]
    imag = float(imag[:-1])
    complexval = complex(real,imag)
    print (complexval)
    

    print (row,col)
    S[row,col]=complexval
    S[col,row]= complex(real, -1.0*imag)

matFile.close()
print (str(np.count_nonzero(S.toarray())))
print ("Start calculation")
start = time.time()
evals_large, evecs_large = eigsh(S,7, which='LM')
end = time.time()
print ("done")
print (evecs_large.shape)
print evals_large
spectrum = open("spec.dat","w+")
for e in evals_large:
    spectrum.write(str(e)+"\n")
spectrum.close()
print end-start
for a in range(0,7):
    stateFileName = "state_"+str(a)+".dat"

    print a
    print evals_large[a]
    print evecs_large[:,a]
    print len(evecs_large[:,a])
    np.savetxt(stateFileName, evecs_large[:,a])

