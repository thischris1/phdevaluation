#!/usr/bin/python3

import numpy as np

try:
    data = np.genfromtxt("test.dat", delimiter='')
except ValueError as e:
    print(e)
    
