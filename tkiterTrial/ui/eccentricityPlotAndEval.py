'''
Created on Feb 3, 2020

@author: chris
'''

if __name__ == '__main__':
    pass


import matplotlib
matplotlib.rcParams['text.usetex'] = True
import pylab as plt
import numpy as np
fileNames = ("eccentricityResultsSmall.dat", "eccentricty_averageValues.dat")
#fileName = "eccentricityResults.dat"
index = 0
for fileName in fileNames:
    data = np.loadtxt(fileName)
#results = np.array([trial,sigma,lcorr,sigmaX,sigmaY,impCount,Vmax])
    sigma = data[:,1]*9.871
#plt.plot(sigma, data[:,2]/100*9.871, '+')

    ecc = data[:,3]/data[:,4]
    print (np.amax(data[:,3]))
    print (np.argmax(data[:,3]), sigma[np.argmax(data[:,3])])
    print (np.argmax(data[:,4]), sigma[np.argmax(data[:,4])])
    print (np.amin(data[:,3:4]))
    print (np.amax(ecc), np.argmax(ecc), sigma[np.argmax(ecc)])
    print (data[np.argmax(ecc),:])
    #plt.plot(sigma,ecc,'o')
    #plt.show()
    data_cut = data[((data[:,4] > 2) & (data[:,4] < 100)& (data[:,3] > 2 ) & (data[:,3]<200))]
    allSigmas = np.unique(data[:,1])
    errors =()
    print (allSigmas)
    for aSigma in allSigmas:
        sigma_cut = data_cut[data_cut[:,1]==aSigma]
        print (len(sigma_cut), np.mean(sigma_cut[:,3]/sigma_cut[:,4]), np.var(sigma_cut[:,3]/sigma_cut[:,4]))
        errors = np.append(errors,(aSigma*9.871,np.mean(sigma_cut[:,3]/sigma_cut[:,4]), np.var(sigma_cut[:,3]/sigma_cut[:,4])))
    print (errors)
    ecc_cut = data_cut[:,3]/data_cut[:,4]
    print(np.mean(ecc_cut))
    fig,ax = plt.subplots()
    if index == 0:
        symbol ='+'
        index = index +1
        plotData = ecc_cut
    else:
        symbol ='o'
        plotData = data_cut[:,2]*9.871
    ax.plot(data_cut[:,1]*9.871,plotData,symbol,linewidth=4, markersize='12')
    #ax.plot(data_cut[:,1]*9.871,errors)
    ax.set_xlabel("$\sigma$ [$l_{0}$] ",fontsize=16)
    ax.set_ylabel("$\sigma_{x}/\sigma_{y}$",fontsize=16)

plt.show()
