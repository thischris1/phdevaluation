'''
Created on Feb 18, 2020

@author: chris
'''
import numpy as np

def read():
    return( readFromFile("spectrum.dat"))
    

def readFromFile(fileName):
    array = np.loadtxt(fileName)
    return array

def calculateGap(data):
    gap = data[3]-data[0]
    return (gap)

def calculateBandwidth(data):
    bw = data[2]-data[0]
    return bw
