'''
Created on Jun 6, 2020

@author: chris
'''
import unittest
import os, sys
from DirectoryResult.ParseDirectory import ParseDirectory

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
class Test(unittest.TestCase):
    def evalDir(self,filehandle,root,name):
         if 'dir_' in name:
            print(os.path.join(root, name))
            a = ParseDirectory(os.path.join(root, name))
            a.printValues(filehandle)

    def testName(self):
        print ("Start")
        startDir = "/home/chris/cluster/clusterResults2/6_18/coul/vosigma7_5e-05"
        filehandle = open("testResults_fehlende_coul_vo_sigma7_e-5.dat", "w+")
        # find all dirs with dir_ in name
        for root, dirs, files in os.walk(startDir, topdown=False):
           # for name in files:
            
            dirCount = len(dirs)
            count = 0
            for name in dirs:
                if 'dir_' in name:
                    print(os.path.join(root, name))
                    try:
                        a = ParseDirectory(os.path.join(root, name))
                        a.printValues(filehandle)
                    except Exception:
                        continue   
                    count = count +1
                    print (count, dirCount)
        filehandle.close()
    def atestSingleDir(self):
        testDir = "/home/chris/cluster/clusterResults2/6_18/coul/vosigma7_5e-05/sigma0035/vo002/dir_10"
        a = ParseDirectory(testDir)
        #filehandle = open("testREsults.dat", "w+")
        #a.printValues(filehandle)
        
        a.printValues()
        self.assertEquals(101,a.evMax[0])
        self.assertAlmostEquals(101,a.evMax[0])
        #filehandle.close()
    def atestPlotrevOverVariance(self):
        data = np.loadtxt("testREsults_6_18_hc_new.dat")
        print (data.shape)
        vosigmas = np.unique(np.around(data[:,4]*data[:,6],decimals=9))
        print(vosigmas)
        for vosigma in vosigmas:
            theData = data[np.where(data[:,4]*data[:,6] == vosigma)]
            labelString = "$V_{0}\sigma = $" + str(vosigma)
            plt.plot(theData[:,12],theData[:,7]*np.sqrt(2*np.pi*18)/1000, '+', ms =10, label = labelString)

        plt.legend()
        plt.xlabel("var V",fontsize=16)
        
        plt.ylabel("$r_{ev} [l_{0}]$",fontsize=16)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.show()
        plt.plot(data[:,12],data[:,7]*np.sqrt(2*np.pi*18)/1000, '+', ms =10)
        plt.xlabel("var V",fontsize=16)
        
        plt.ylabel("$r_{ev} [l_{0}]$",fontsize=16)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.show()
                    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
