#!/usr/bin/python


import string
import multiprocessing as mp
import os, glob
import shutil
import numpy as np
resultDir = os.path.join(os.getcwd(),"results")
sourceDir ="."
def findPosFromFileHeader(fileName):
    file = open(fileName,"r")
    firstLine = file.readline().rstrip()
#    print firstLine
    if "postion" not in firstLine:
        return "Unknown position", "Unknown position"
    else:
        # split and take last two elements
        parts = firstLine.split(' ')
        length = len(parts)
        y = parts[length-1]
        x = parts[length-2]
        retVal = str(x)+"  "+ str(y)
        file.close()
        return x,y

def findMaximumInFile(fileName, limit, column):
    # load file into np array
    fileContent = np.loadtxt(fileName)
    fileContent = fileContent[:limit]
    maxCol = fileContent[:,column]
    posOfMax = np.argmax(maxCol)
    valOfMax = np.amax(maxCol)
    return (posOfMax, valOfMax)


def calculateConditionalCorrelation(x,y,state, basis, MCSteps):
    # create sampleElectron.dat

    writeSampleElectrons(x,y)
    print state
    execString = "correlation -s "+str(state)+ " -b "+str(basis)+ " -n "+str(MCSteps)
    print execString
    os.popen(execString)
    # save results
    targetFile = "fixedElecVortexDistra_x_"+str(x)+"_y_"+str(y)+".dat"
    copyCommand = 'cp fixedElecVortexDistra.dat '+str(targetFile)
    os.popen(copyCommand)


def writeSampleElectrons(x,y):
    sampleFileName = "./sampleElectrons.dat"
    sampleFile = open(sampleFileName,"w")
    sampleFile.write(str(x)+str("   ") + str(y))
    sampleFile.close()
    

def cleanUp(dir):
    print "Cleaning up "+str(dir)
    os.popen ("rm -rf "+dir)
    return


def saveResults(fileExtension):
    for file in glob.glob("*orrel*"):
        print file
        # rename File (split before dat)
        filename, file_extension = os.path.splitext(file)
        copyCommand = "cp "+file+" "+resultDir+"/"+filename+fileExtension+".dat"
        
def gaussian(Vo,sigma, correl):
    
    procNr = os.getpid()
    dirpath = os.getcwd()
    print "exec dir" + dirpath
    fileExtension = "strength_"+str(Vo)+"-wx-"+str(sigma)+"-wy-"+str(sigma)+".dat"
    print fileExtension
    dirPath = "test"+str(procNr)
    if (os.path.exists(dirPath)== False):
        # do something
        os.mkdir(dirPath)
    copyCommand = 'cp gaussian.par '+str(dirPath)
    os.popen(copyCommand)
    os.chdir(dirPath)
    
    gaussFile = open("impurities.dat","w")
    gaussFile.write("0.5 \t 0.5 "+str(sigma)+"\t "+str(sigma)+"\t " + str(Vo)+ "\n")
    gaussFile.close()
    os.system("/home/chris/devel/gaussian/gaussian")
    # get state file (the one with .dat0 at the end)
    statePathSearch = "state*.dat0"
    for file in  glob.glob("*.dat0"):
        print file
        statePath = file
    specFile = "spectrum_Vo_"+str(Vo)+"_wx_"+str(sigma)+".dat"
    print "state" + str(statePath)
    copyCommand2 = ' cp spectrum.dat' + ' ' + specFile
    os.popen(copyCommand2)
    if correl == False:
        os.chdir('..')
        return
    # now for correlation
    correlationCommand = "correlation -s "+statePath + " -b bs4-12.dat -n 10"
    print correlationCommand
    os.system(correlationCommand)
    # save results
    saveResults(fileExtension)
    os.chdir('..')
    cleanUp(dirPath)
    return 1;

def conditionalCorrelation(x,y,state, basis, MCSteps):
    # create sampleElectron.dat
    writeSampleElectrons(x,y)    

    execString = "correlation -s "+str(state)+ " -b "+str(basis)+ " -n "+str(MCSteps)
    print execString
    os.popen(execString)
    # save results
    targetFile = "fixedElecVortexDistra_x_"+str(x)+"_y_"+str(y)+".dat"
    copyCommand = 'cp fixedElecVortexDistra.dat '+str(targetFile)
    os.popen(copyCommand)




def parseSpectrumFile (fileName):
    file = open(fileName, 'r')
    print ("File is open")
    line =""
    groundState = 0.0
    cnt = 0
    for line in file:
#        line = file.readline()
        print (line)
        if cnt == 0:
            groundState = float(line)
            print (groundState)
            cnt +=1
        else:
            if cnt == 3:
                gap = float(line)-groundState
                cnt +=1
    file.close()
    return (groundState)
