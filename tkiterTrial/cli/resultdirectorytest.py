'''
Created on Feb 16, 2020

@author: chris
'''
import unittest
import os
import ResultDirectory as rd

import CorrelationFit
from DirectoryResult import ParseDirectory

class Test(unittest.TestCase):
    

    def atestParseDir(self):
        ad = ParseDirectory.ParseDirectory("./test/dir_1")
        self.assertEqual(ad.ne,'6',"")
        self.assertEqual(ad.Nm,'18',"" )
        self.assertEqual(ad.interaction,"HC")
        self.assertEqual(0.01, ad.sigma, "")
        self.assertEqual(0.0013, ad.Vmax, "")
        self.assertEqual(-0.0013, ad.Vmin, "")
        file = open("./EinFile.dat","w")
        ad.printValues(file)
        file.close()
        
        

    def atestParseDir2(self):
        ad = ParseDirectory.ParseDirectory("/home/chris/eclipse-workspace/tkiterTrial/cli/test/dir_2")
        self.assertEqual(ad.ne,'5',"")
        self.assertEqual(ad.Nm,'13',"" )
        self.assertEqual(ad.interaction,"Coulomb")
        file = open("./EinAnderesFile.dat","w")
        ad.printValues(file)

    def yatestName(self):
        directoryName = "testDirectory/test/dir_1"
        aDir = rd.resultDirectory(directoryName)
        self.assertEqual(6, rd.electronNumber(), "Msg")
        aDir.printValues()
       

    def yatestFit(self):
        fileName = "test/dir_1/evCorrelation_1.dat"
        
        results= CorrelationFit.fitAndPlot2(fileName,True,plot = False)
        print (results)
        max = results[0]
        print (max)
    def testDirectory(self):
        directoryName="/home/chris/cluster/clusterResults2/6_18/hardcore/vosigma4_e_05/sigma0075"
        aDir = rd.resultDirectory(directoryName)
        resFile = open("Results_6_18_hc_sigma0075.dat","w+")
        aDir.printValues(resFile)
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()