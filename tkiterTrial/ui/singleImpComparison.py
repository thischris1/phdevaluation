'''
Created on Jul 22, 2021

@author: root
'''

import cli.Gaussian as gauss
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import rc
sigma = np.linspace(0, 0.15,10)
Vo = np.linspace(-1.0,1.0,10)



x = np.linspace(0,1,101)
y = np.linspace(0,1,101)

xx,yy =np.meshgrid(x,y)

for asigma in sigma:
    for aVo in Vo:
        pot = gauss.getPotential([0.5,0.5,aVo], asigma, xx, yy)
        pot2 = gauss.getPotential([0.51,0.51,-1.0*aVo], asigma, xx, yy)
        print (pot.shape)
        print (pot[0,0])
        potMean = np.mean(pot)
        potvariance = np.var(pot)
        print ("Sigma, vo",asigma,aVo)
        print ("Mean", potMean, np.mean(pot2))
        print ("Variance", potvariance, np.var(pot2), np.var(pot+pot2))
        print ("Max values", np.max(pot), np.min(pot2))
