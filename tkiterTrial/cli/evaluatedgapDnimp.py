
#!/usr/bin/python3

import numpy as np
import sys, os
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

data = np.loadtxt("resultFileNimpvaried.dat")

allNimps=np.unique(data[:,0])
results=[]
for nimp in allNimps:
    subarray = data[np.where(data[:,0]==nimp)]
    gapAverage = np.mean(subarray[:,5])
    gapVar = np.var(subarray[:,5])
    resTuple =  (nimp, gapAverage,gapVar)
    print (nimp, gapAverage,gapVar,subarray.shape)
    results.append(resTuple)
resArray = np.asarray(results)
print (resArray)
plt.plot(data[:,0],data[:,5],"ro")
plt.plot(resArray[:,0], resArray[:,1],"b+")
plt.show()
diff = np.diff(resArray[:,1])/np.diff(resArray[:,0])
print (diff)

