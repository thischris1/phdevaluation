'''
Created on Jun 6, 2020

@author: chris
'''
import unittest
import os, sys
from DirectoryResult.ParseDirectory import ParseDirectory

class Test(unittest.TestCase):


    def testName(self):
        print ("Start")
        startDir = "/home/chris/cluster/clusterResults2/6_18/hardcore/4800imps/vosigma4_e_05/sigma0025"
        # find all dirs with dir_ in name
        for root, dirs, files in os.walk(startDir, topdown=False):
           # for name in files:
           #     print(os.path.join(root, name))
            for name in dirs:
                if 'dir_' in name:
                    print(os.path.join(root, name))
        
        p=os.listdir(r'/home/chris/cluster/clusterResults2/6_18/coul/')
        for i in p:
            print (i)
            if os.path.isdir(i):
                print(i)
 
    def testSingleDir(self):
        testDir = "/home/chris/cluster/clusterResults2/6_18/hardcore/4800imps/vosigma4_e_05/sigma0075"
        a = ParseDirectory(testDir)
        a.printValues("testResultshcvosigma4e05_0075.dat")
                    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()