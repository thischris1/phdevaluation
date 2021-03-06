

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
    print (state)
    execString = "correlation -s "+str(state)+ " -b "+str(basis)+ " -n "+str(MCSteps)
    print (execString)
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
    print ("Cleaning up "+str(dir))
    os.popen ("rm -rf "+dir)
    return


def saveResults(fileExtension):
    for file in glob.glob("*orrel*"):
        print (file)
        # rename File (split before dat)
        filename, file_extension = os.path.splitext(file)
        copyCommand = "cp "+file+" "+resultDir+"/"+filename+fileExtension+".dat"
        
def gaussian(Vo,sigma, correl):
    
    procNr = os.getpid()
    dirpath = os.getcwd()
    print ("exec dir" + dirpath)
    fileExtension = "strength_"+str(Vo)+"-wx-"+str(sigma)+"-wy-"+str(sigma)+".dat"
    print (fileExtension)
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
        print (file)
        statePath = file
    specFile = "spectrum_Vo_"+str(Vo)+"_wx_"+str(sigma)+".dat"
    print ("state" + str(statePath))
    copyCommand2 = ' cp spectrum.dat' + ' ' + specFile
    os.popen(copyCommand2)
    if correl == False:
        os.chdir('..')
        return
    # now for correlation
    correlationCommand = "correlation -s "+statePath + " -b bs4-12.dat -n 10"
    print (correlationCommand)
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
    print (execString)
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
                                                                                                                                                                                                                                                                                                                                                                                                       CorrelationFit.py                                                                                   0000664 0001750 0001750 00000004505 13622545457 013401  0                                                                                                    ustar   chris                           chris                                                                                                                                                                                                                  #!/usr/bin/python

from numpy.random import uniform
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy import interpolate
from scipy.optimize import curve_fit
import pylab as p
import sys
from scipy.interpolate import interp1d
import os.path 
files = []

fitStart = -1
fitEnd = 1000
max = -10
def aPower(x,a,b,c):
    retVal = c*np.power(x,a)+b
    return retVal

def quadraticOnly(x,a,b):
    retVal = a*x*x+b
    return retVal


def checkFitPoints(fitStart, fitEnd,maxPos):
    if (fitEnd - fitStart > 3):
        #passt
        return fitStart, fitEnd
    else:    
        # zu dicht beieinander
        print ("Fix fitintervall")
        if fitEnd < 3:
            fitEnd = maxPos -1
            fitStart = 2
            return fitStart, fitEnd


def fitAndPlot2(fileName, normalize, plot =False):
    
    print (fileName)
    
    corrFile=np.loadtxt(fileName)
    
  
    print ("First in corrfile x",corrFile[0,0])
    r = corrFile[:,0]
    evCorr = corrFile[:,1]
    r = r[:400]
    evCorrTemp = evCorr[:400]
    if np.argmax(evCorrTemp) < 1.5:
        evCorr = np.delete(evCorrTemp,[0])
        r = np.delete(r,[0]) 
        print ("Delete an element")
    else:
        evCorr = evCorrTemp
    
    
    if normalize == True:
        r = r+1
        evCorr = evCorr/r
    
    maxPos = np.argmax(evCorr)
    maxCorrVal = np.amax(evCorr)
    maxCorrVal = evCorr[maxPos]
    print ("MAximum value is " + str(maxCorrVal) + " at " + str(maxPos))
    print (maxPos)
    leftVal = maxPos
    rightVal = maxPos
    # look for threshold value
    threshold = maxCorrVal*0.75
    # erst nach links
    for index in  range(maxPos):
        print (str(maxPos-index))
        if evCorr[maxPos-index] > threshold:
            continue
        else:
            leftVal = maxPos-index
            break
    
    print ("Left end is " + str(leftVal))
     # now for right... 
    for index in range(3*maxPos):
        if evCorr[maxPos+index] > threshold:
            continue
        else:
            rightVal = maxPos+index
            break
    
    print ("Right value is " +str(rightVal))
        
    width = rightVal -leftVal
    print ("Width = " + str(width))
    return (maxPos, width)
                                                                                                                                                                                           evaluatedgapDnimp.py                                                                                0000664 0001750 0001750 00000001600 13632122444 014064  0                                                                                                    ustar   chris                           chris                                                                                                                                                                                                                  
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

                                                                                                                                evCorrelationvslcorrplots.py                                                                        0000664 0001750 0001750 00000023322 13647247444 015766  0                                                                                                    ustar   chris                           chris                                                                                                                                                                                                                  '''
Created on Apr 9, 2020

@author: chris
'''
import numpy as np
import sys, os
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib import rc
from sympy.physics.quantum.sho1d import ad
from sessioninstaller.core import ConfirmInstallDialog


rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


# berechne den mittelwwet und die Varianz der Spalte index in data
# gib den Mittelwert und die Varianz zurueck
def calculateMeanAndVarianceSingleEntity( data, index):
    tempArray = []
    if data.size == 0:
        return
    print (data.shape)
    for aDatum in data:
        tempVal = aDatum[index]
        if np.isnan(tempVal):
            continue
        tempArray= np.append(tempArray,tempVal)
    
    tempArray = np.asarray(tempArray)
    mean = np.mean(tempArray)
    var = np.var(tempArray)
    if np.isnan(mean) or np.isnan(var):
        return
    return (mean,var)

# Berechne varainz und mittelwert in den Spalten in indexlist von data  
# Gibt ein Array der Form [[mean1, var1],[mean2,var2],...] zurueck
#
def calculateMeanAndVarianceListOfEntities( data, indexlist):
    retArray = []
    for index in indexlist:
        (mean, var) = calculateMeanAndVarianceSingleEntity(data, index)
        retArray = np.hstack((retArray,[mean,var]))
    return retArray

def calculateMeanFormVmaxSigmaList( vosigmaList, data, indexList, plot = True):
    # indices to be meaned and aard given in indexList,
    # 
    indexSize = int(len(indexList))*2 +1
    resultArray = np.empty([0,indexSize])
    exVoSigmaList = []
    print (resultArray.shape)
    for vosigma in vosigmaList:
        vosigmaConstArray = data[np.where(np.abs(data[:,VmaxIndex]*data[:,sigmaIndex] - vosigma) < 1e-07)]
        if vosigmaConstArray.size == 0:
         
            continue
        exVoSigmaList.append(vosigma)
        retVal = calculateMeanAndVarianceListOfEntities(vosigmaConstArray, indexList)
        retVal= np.append(retVal,vosigma)
        print (retVal.shape)
        print (resultArray.shape)
        resultArray = np.vstack((resultArray,retVal))
    if plot:
        plt.plot(exVoSigmaList, resultArray[:,0], "ro")
        plt.xlabel("$V_{max}\sigma$")
        plt.ylabel("$r_{ev}$")
        plt.show()

def plotrevvvOverlcorrAllVoSigma(data, vosigmaList):
    
    print (data.shape)
    for vosigma in vosigmaList:
        vosigmaConstArray = data[np.where(np.abs(data[:,VmaxIndex]*data[:,sigmaIndex] - vosigma) < 1e-09)]
        print (vosigmaConstArray.shape)
        if vosigmaConstArray.size == 0:
         
            continue
        titleString ="V  sigma = "+ str(vosigma)
        plt.plot(vosigmaConstArray[:,lcorrIndex],vosigmaConstArray[:,evMaxIndex], "ro", label=titleString)
        
        plt.xlabel("l_{corr}")
        plt.ylabel("r_{ev}")
        plt.legend(titleString)
    plt.show()

def addMaxtwoColumn(data):
    maxArray =[]
    for aData in data:
        
        
        if aData[twodcorrelationx]> aData[twodcorrelationy]:
            maxArray = np.append(maxArray,aData[twodcorrelationx])
        else:
            maxArray= np.append(maxArray,aData[twodcorrelationy])
    maxArray = np.expand_dims(maxArray, -1)
    print (maxArray.shape)
    return (maxArray)
def gnuplotComparisonPlot(data, Ne, ia):
# plot Ne = 5 , Interaction = HC, all points

    
    electronNumberDataArray =allData[allData[:,0]==Ne]  
    HCArray = electronNumberDataArray[electronNumberDataArray[:,interactionIndex]==ia]
    vosigma = electronNumberDataArray[:,sigmaIndex]*electronNumberDataArray[:,VmaxIndex]
    vosigmalList = np.unique(vosigma.round(decimals = 6))
    print (vosigmalList)
    figs,axs = plt.subplots(2,2 )
    titleString = " Electrons"
    titleString = str(Ne)+titleString
    if ia == 0:
        titleString = titleString+" Coulomb interaction"
    else:
        titleString = titleString+" SRI"
    plt.suptitle(titleString)   
    
   # axs[0,0].plot(HCArray[:,lcorrIndex], HCArray[:,evMaxIndex], 'o')
   # axs[0,1].plot(HCArray[:,lcorrIndex], HCArray[:,vvMaxIndex]*0.5,'+')
    for aVmaxSigma in vosigmalList:
        vosigmaConstArray = HCArray[np.where(np.abs(HCArray[:,VmaxIndex]*HCArray[:,sigmaIndex] - aVmaxSigma) < 1e-09)]
        sigmaList = np.unique(vosigmaConstArray[:,sigmaIndex])
        print (sigmaList)
        
        if len(sigmaList) > 3:
            correlationVsSigmaArray = np.empty((0,7))
            max = 0.0
            for aSigma in sigmaList:
                sigmaData = vosigmaConstArray[vosigmaConstArray[:,sigmaIndex] == aSigma]
                print (sigmaData.shape)
                meanRev = np.nanmean(sigmaData[:,evMaxIndex])
                #meancorrMax = np.mean(np.maximum(sigmaData[:,twodcorrelationx],sigmaData[:twodcorrelationy]))
               # print (meancorrMax)
                meanrvv = np.nanmean(sigmaData[:,vvMaxIndex]) * 0.5
                varRev = np.sqrt(np.nanvar(sigmaData[:,evMaxIndex],dtype=np.float64))
                varRvv = np.sqrt(np.nanvar(sigmaData[:,vvMaxIndex], dtype=np.float64))
                meantwod = np.nanmean(sigmaData[:,twodmaxIndex])
                vartwod = np.nanvar(sigmaData[:,twodmaxIndex])
                if np.isnan(varRev):
                    print ("NAN in VAR")
                    print  (sigmaData[:,evMaxIndex])
                    print ("ENDE")
                print ("variance = ", varRev, varRvv, vartwod,aSigma)
                print ("mean = " ,meanRev, meanrvv, meantwod,aSigma)
                # einpacken in ein array
                resultArray = np.array([aSigma,meanRev,varRev,meanrvv,varRvv,meantwod,vartwod])
                correlationVsSigmaArray = np.vstack((correlationVsSigmaArray, resultArray)) 
            print (correlationVsSigmaArray.shape)
            if max < np.amax(correlationVsSigmaArray[:,1])*1.2:
                max = np.amax(correlationVsSigmaArray[:,1])*1.2
            #axs[2,0].plot(vosigmaConstArray[:,sigmaIndex]*9.871, vosigmaConstArray[:,evMaxIndex], 'ro')
            axs[0,0].errorbar(correlationVsSigmaArray[:,0]*9.871,correlationVsSigmaArray[:,1],yerr=correlationVsSigmaArray[:,2], marker='s',
          ms=20, mew=4,linestyle='None',label =str(aVmaxSigma))
            axs[0,0].set_title("$r_{ev}$")
            axs[0,0].legend(numpoints=1)
            axs[0,0].set_ylim(bottom=0)
            axs[0,0].set_ylim(top=max)
           # axs[2,1].plot(vosigmaConstArray[:,sigmaIndex]*9.871, vosigmaConstArray[:,vvMaxIndex], 'bo')  
            axs[0,1].errorbar(correlationVsSigmaArray[:,0]*9.871,correlationVsSigmaArray[:,3],yerr=correlationVsSigmaArray[:,4], marker='s',
          ms=20, mew=4,linestyle='None',label=str(aVmaxSigma))
            axs[0,1].set_title("$r_{vv}$")
            axs[0,1].set_ylim(bottom=0)
            axs[0,1].set_ylim(top = max)
            axs[0,1].legend(numpoints=1)
            axs[1,0].errorbar(correlationVsSigmaArray[:,5]*9.871, correlationVsSigmaArray[:,1],xerr=correlationVsSigmaArray[:,6], yerr=correlationVsSigmaArray[:,2],  marker='s',
          ms=20, mew=4,linestyle='None',label=str(aVmaxSigma))
            axs[1,0].set_ylim(bottom=0)
            axs[1,0].set_ylim(top = max)
            axs[1,0].set_xlim(left=0)
            axs[1,0].legend(numpoints=1)
            axs[1,0].set_title("$r_{ev}(l_{corr}^{2d})$")
        else:
            continue  
        labelString = str(aVmaxSigma)
      #  axs[1,0].plot(vosigmaConstArray[:,lcorrIndex], vosigmaConstArray [:,evMaxIndex], '*', label = labelString)
        axs[1,1].plot(vosigmaConstArray[:,lcorrIndex], vosigmaConstArray [:,vvMaxIndex], '+', label = labelString)
    plt.legend()
   
    plt.show()
    #sys.exit()
    
# gnuplot indices 
NeIndex = 0  #1
NmIndex = 1 #2
interactionIndex = 3 #4 
impCountIndex = 4  #5
VmaxIndex = 5 #6
VminIndex = 6  #7
sigmaIndex = 7 #8
lcorrIndex = 8  #9
twodcorrelationx = 9 #10
twodcorrelationy = 10 #11
evMaxIndex = 11 # 12
vvMaxIndex = 12 # 13
gsIndex = 13  #14
gapStateIndex = 16  #17
twodmaxIndex = 19
fileName  = "results_all_cluster.dat"
allData = np.genfromtxt(fileName)
print  (allData.shape)
extraColumn = addMaxtwoColumn(allData)
print (extraColumn.shape)

allData = np.append(allData,extraColumn,axis=1)
print (allData.shape)
print (allData[0,twodcorrelationx], allData[0,twodcorrelationy], allData[0,twodmaxIndex])
nelist = np.unique(allData[:,0])
print (nelist)
ialist = [1,0]
#print (vmaxsigmalist)
ialist = np.unique(allData[:,interactionIndex])
gnuplotComparisonPlot(allData,5,1)
gnuplotComparisonPlot(allData,6,1)
sys.exit()
for ne in nelist:
    print(ne)
    electronNumberDataArray =allData[allData[:,0]==ne]
    for ia in ialist:
        print (ia)
        subDataArray = electronNumberDataArray[electronNumberDataArray[:,interactionIndex]==ia]
        plt.plot(subDataArray[:,8],subDataArray[:,evMaxIndex], 'o')
        plt.show()
        vosigma = subDataArray[:,VmaxIndex]*subDataArray[:,sigmaIndex]
        vmaxsigmalist = np.unique(vosigma.round(decimals = 6))  
        print (vmaxsigmalist)
        print (subDataArray[0,:])
        #print (subDataArray[1223,:])
        
        plotrevvvOverlcorrAllVoSigma(subDataArray, vmaxsigmalist)
        continue
        #calculateMeanFormVmaxSigmaList(vmaxsigmalist, subDataArray, (evMaxIndex,vvMaxIndex), True)
        #print (subDataArray)
        #print ("====================================================")
        plt.plot(subDataArray[:,VmaxIndex]*subDataArray[:,sigmaIndex],subDataArray[:,lcorrIndex], "ro")
        plt.plot(subDataArray[:,VmaxIndex]*subDataArray[:,sigmaIndex], subDataArray[:,twodcorrelationx],"bo")
        plt.title(str(int(ne))+" electrons, interaction = "+ str(int(ia)))
        plt.show()
        plt.plot(subDataArray[:,gsIndex]-subDataArray[:,gapStateIndex],subDataArray[:,lcorrIndex], "ro")
        plt.title(str(ne)+" electrons")
        plt.show()                                                                                                                                                                                                                                                                                                              Gaussian.py                                                                                         0000664 0001750 0001750 00000027505 13643616731 012231  0                                                                                                    ustar   chris                           chris                                                                                                                                                                                                                  '''
Created on Sep 15, 2019

@author: chris
'''
import numpy as np
import os,sys
from scipy import signal
from scipy.optimize import curve_fit
import scipy.optimize as opt
import Spectrum
    
def getPotential(imp, sigma, x,y):
    strength = imp[2]   
    sigsquare = sigma*sigma
    posx = imp[0]
    posy= imp[1]
    
    exponent = (posx-x)*(posx-x)+(posy-y)*(posy-y)
    exponent = exponent/sigsquare*-1.0
        
    return (strength*np.exp(exponent))

def getPotentialSum(imps,sigma,x,y):
    retVal = 0.0
    for anImp in imps:
        retVal = retVal + getPotential(anImp,sigma,x,y)
    return retVal
    
# returns the width and the positions of the impurities
def readGaussianFile(fileName):
    gaussArray = np.loadtxt(fileName, usecols=(0,1,2,4))
    impsx = gaussArray[:,0]
    impsy = gaussArray[:,1]
    sigma = gaussArray[0,2]
    strengths = gaussArray[:,3]
    
    imps = np.delete(gaussArray, 2, 1)
    return imps, sigma

def createPotentialFromGaussianFile(fileName):
    imps,sigma = readGaussianFile(fileName)
    x = np.linspace(0.0,1.0,101)
    y = np.linspace(0.0,1.0,101)
    x,y = np.meshgrid(x,y)
    pot = getPotentialSum(imps, sigma, x, y)
    return x,y,pot
    
def readPotentialFile(fileName):
    data = np.loadtxt(fileName)
    x = data[:,0]
    y = data[:,1]
    pot = data[:,2]
    x,y =np.meshgrid(x,y)
    return (x,y,pot)
# get , potentialvalue at x,y from file  
def getPotentialFromFile(fileName, xPos, yPos):
    x,y,potential = readPotentialFile(fileName)
    xSize = len(x)
    ySize = len(y)
    xIndex = int(xPos*xSize)
    yIndex = int(yPos*ySize) 
    return potential[xIndex,yIndex]

def generateRandomPotential(impCount, Vmax, sigma, Debug=False):

    if Debug == True:
       print ("generate a random potential with "+str(impCount)+" impurities")
    x =  np.arange(101)
    y = np.arange(101)
    x,y = np.meshgrid(x,y)
    imps = np.random.rand(impCount,2)
    strengths = np.random.rand(impCount,1)-0.5
    strengths = strengths*Vmax

    fimps = np.hstack((imps,strengths))
    potential = getPotentialSum(fimps,sigma,x/100.0,y/100.0)
   # print (potential)
#    potential = np.reshape(101,101)
    return strengths, fimps, potential

def procedure(i, impCount, Vmax,sigma, returnPotential=False):
    # print (i)
    path = "dir_"+str(i)
    # check if dir exists ( 1 ... numCount
    if (not os.path.isdir(path)):
       #do nothing
       os.mkdir(path)
    # change into dir
    os.chdir(path)
    createGaussian()
    #shutil.copy("../gaussian.par", ".")
    # create random.dat 
    writeRandom(impCount, sigma, Vmax)
   
    # create if necessary 
   
    # run gaussian -p
    os.system("gaussian -p")
    # evaluate
    
    data = readPotentialFile("PotentialArray.dat")
    pot = data[2]
    #print ("PotentialGroesse")
    #print (len(pot))
    lcorr = calculateAutoCorrelationFromFile("PotentialArray.dat")
    
    os.chdir("..")
    #artMean, artVariance,artMax = Gaussian.createAndEvaluatePotential(Vmax, impCount, sigma, x, y)
    #print ("Artifical")
    #print (artMean, artVariance, artMax)
   
    # print (np.mean(pot),np.var(pot),np.amax(pot))
    if returnPotential == True:
        return (pot,lcorr)
    else:
        return (np.mean(pot),np.var(pot),np.amax(pot), lcorr)
    
def createGaussian(coulomb = False):
    f = open("gaussian.par","w")
    f.write ("./bs# root-name of basis file \n ./state_5_15_hc# root-name of vector file \n ./dnY0.0# root-name of density file \n")
    f.write ("./ldY0.0# root-name of landau-diagonal file\n")
    f.write ("./pot# root-name of potential file\n")
    f.write ("5       # Ne: Nr. of electrons\n")
    f.write ("15              # Nm: Nr. of flux quanta (i.e. Ne/Nm=filling factor)\n")
    f.write ("0               # spinYes: 0=spin polarized, 1=not necessarily sp. pol.\n")
    f.write ("0               # reqSz: dtto, with total Sz (applies only if spinYes=1)\n")
    f.write ("2               # mat_type: 0=FULL_REAL, 1=SPARSE_REAL, 2=FULL_CPLX\n")
    f.write ("1               # type of vector-file to generate: 0->ascii, 1->raw binary\n")
    f.write ("7              # eigsToFind: Nr. of eigvals/eigvecs to be found\n")
    f.write ("1.0             # a: size of the system (vert.)\n")
    f.write ("1.0       # b: size of the system (horiz.)\n")
    f.write ("0.0             # bli: related to finite thickness\n")
    f.write ("0               # type of barrier potential: 0 -> gaussian, 1 -> delta\n")
    if coulomb:
        f.write ("0               # type of e-e interaction: 0 -> Coulomb, 1 -> hardcore\n")
    else:
        f.write ("1               # type of e-e interaction: 0 -> Coulomb, 1 -> hardcore\n")
    f.write ("-2.0            # energy-offset\n")
    f.write ("0.0             # flux of solenoid1 in units of h/e\n")
    f.write ("0.0             # flux of solenoid2 in units of h/e\n")
    f.write ("100             # xkmax: Sum from -kmax to kmax for Barrier in x-direction (resp. hole)\n")
    f.write ("100             # ykmax: Sum from -kmax to kmax for Barrier in in y-direction (r\n")
    f.write ("random.dat")
    f.close()
                 
def writeRandom(impCount,sigma, strength):
    negstrength = -1.0*strength
    f = open("random.dat", "w")
    f.write(str(impCount)+"\n")
    f.write (str(sigma)+"\n")
    f.write (str(strength)+"\n")
    f.write (str(negstrength)+"\n")
         
    f.close()

def readRandom(fileName):
    if os.path.exists(fileName):
        data = np.loadtxt(fileName)
        return data[0], data[1],data[2], data[3]

def runGaussian(mode):
    if 'p' in mode:
        os.system("gaussian -p")
    else:
        os.system("gaussian")

def createPotentialFromGaussian(i, impCount, sigma, Vmax, gaussMode ='p', interaction ='HC'):

    path = "dir_"+str(i)
    # check if dir exists ( 1 ... numCount
    if (not os.path.isdir(path)):
       #do nothing
       os.mkdir(path)
    # change into dir
    os.chdir(path)
    if 'HC' in interaction:
        createGaussian()
    else:
        createGaussian(True)
        #shutil.copy("../gaussian.par", ".")
    # create random.dat 
    writeRandom(impCount, sigma, Vmax)
   
    # create if necessary 
   
    # run gaussian -p
    runGaussian(gaussMode)
    
    # evaluate
    
    data = readPotentialFile("PotentialArray.dat")
    pot = data[2]
    potSize = int(np.sqrt(len(pot)))
    #print (potSize)
    x = np.arange(potSize-1)
    y = np.arange(potSize-1)
    x = x/float(potSize-1)
    y = y/float(potSize-1)
    x = np.append(x,1.0)
    y = np.append(y,1.0)
    

    (x,y) = np.meshgrid(x,y)
    pot = np.reshape(pot,(potSize,potSize))
    #print (x)
    
    if not 'p' in gaussMode:
        # read spectrum
        spectrum = Spectrum.read()
        
        return x,y,pot,spectrum
    return x,y,pot

def calculateAutoCorrelationFromFile(fileName):
    pot = readPotentialFile(fileName)[2]
    # resize pot
    #print (pot.shape)
    potSize = len(pot)
    #print (potSize)
    potSize = int(np.sqrt(potSize))
    #print (potSize)
    pot = np.reshape(pot,(potSize,potSize))
    return calculateAutoCorrelation(pot, False)

def calculateAutoCorrelationData(pot):
    corr = signal.correlate2d(pot,pot)
    #    extract middle x
    corrSize = len(corr)
    cutX = corr[int(corrSize*0.5),:]
    cutLength = len(cutX)
    x = np.arange(corrSize)
    potX = np.arange(0,corrSize,10000)

   # print (cutX.shape)
    # fit to Gaussian
    p0 = [1., corrSize*0.5, 1.]
    coeff, var_matrix = curve_fit(gauss, x, cutX, p0=p0)
    fitted = gauss(x,*coeff)
    return (corr, cutX, np.abs(coeff[2]),fitted, np.sqrt(np.abs(var_matrix[2,2])))


def calculateAutoCorrelation(pot, plotting = False):
    corr = signal.correlate2d(pot,pot)
    #    extract middle x
    corrSize = len(corr)
    cutX = corr[int(corrSize*0.5),:]
    x = np.arange(corrSize)
    potX = np.arange(0,corrSize,10000)

   # print (cutX.shape)
    # fit to Gaussian
    p0 = [1., corrSize*0.5, 1.]
    coeff, var_matrix = curve_fit(gauss, x, cutX, p0=p0)
    fitted = gauss(x,*coeff)
    
#    fitted = gauss(potX,*coeff)
    if plotting ==True:
        import matplotlib.pyplot as plt
        plt.plot(x,cutX, label ='data')
        plt.plot(x,fitted, label='Fitted', linewidth=3)
        plt.legend()
        plt.show()
    #print (coeff)
    return np.abs(coeff[2])

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

    
def findSmallestAndLargestCorrelationLength( Vmax, Nimp, sigma, reps):
    potSmallLcorr = None
    potLargeLcorr = None
    lcorrSmall = 1000.0
    lcorrLarge = -1000.0
    for index in np.arange(0,reps):
        (pot,lcorr) = procedure(index,Nimp,Vmax,sigma,True)
        # start 
        if index == 1:
            potSmallLcorr = pot
            potLargeLcorr = pot
            lcorrSmall = lcorr
            lcorrLarge = lcorr
            continue
        if lcorrSmall > lcorr:
            potSmallLcorr = pot
            lcorrSmall = lcorr
            continue
        if lcorrLarge < lcorr:
            potLargeLcorr = pot
            lcorrLarge = lcorr
    return (potLargeLcorr, lcorrLarge,potSmallLcorr, lcorrSmall)


def calculateAutoCorrelation2dFromFile(fileName, plot =False):
    data  = np.loadtxt(fileName)
    pot = data[:,2]
    potSize = len(pot)
    print (potSize)
    potSize = int(np.sqrt(potSize))
    print (potSize)
    pot = np.reshape(pot,(potSize,potSize))
    return (calculateAutoCorrelation2dFromPotentialData(pot, plot))
   
    
def calculateAutoCorrelation2dFromPotentialData(pot, plot = False):
    
    corr = signal.correlate2d(pot,pot).ravel()
    return calculateAutoCorrelation2dFromACFplot(corr, plot)

def calculateAutoCorrelation2dFromACFplot(corr, plot = False):
    
    corrSize = int(len(corr))
    print ("Corr size ="+str(corrSize))

# Create x and y indices
    x = np.linspace(0, 200, 201)
    y = np.linspace(0, 200, 201)
    x, y = np.meshgrid(x, y)

    passTuple = (x,y)
#create data
    data = twoD_Gaussian(passTuple,3, 100, 100, 20, 40, 0, 10)
    initial_guess = (3,100,100,20,40,0,10)

#data_noisy = data + 0.2*np.random.normal(size=data.shape)
    data_noisy = corr
    
    

    popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data_noisy, p0=initial_guess)
    data_fitted = twoD_Gaussian((x, y), *popt) 
#print (popt, pcov)
# plot twoD_Gaussian data generated above
    if (plot == True): 
        import pylab as plt
       
        fig, ax = plt.subplots(1, 1)
        ax.hold(True)
        ax.imshow(data_noisy.reshape(201, 201), cmap=plt.cm.jet, origin='bottom',extent=(x.min(), x.max(), y.min(), y.max()))
        ax.contour(x, y, data_fitted.reshape(201, 201), 1, colors='w')
#ax.colorbar()
        plt.show()
    sigmax = np.abs(popt[3])
    sigmay = np.abs(popt[4])
    print (sigmax,sigmay)
    return (sigmax, sigmay,data_fitted)

def twoD_Gaussian(xyTuple,amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x = xyTuple[0]
    y = xyTuple[1]
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)  + c*((y-yo)**2)))
    return g.ravel()

def readSpectrumFile(fileName):
    retVal = ""
    if os.path.exists(fileName):
        
        data = np.loadtxt(fileName)
        for val in data:
            retVal = retVal +" "+ str(val)
        return retVal



    
def readGaussianPar(fileName):
    
    
    # read all ato once into array
    lineList = [line.rstrip('\n') for line in open(fileName)]
    
    # line 5 Ne line 6 Nm
    Neline = lineList[5]
    NmLine = lineList[6]
    iaLine = lineList[16]
    parts = Neline.split('#')
    Ne = parts[0].rstrip()
    parts = NmLine.split('#')
    Nm = parts[0].rstrip()
    parts = iaLine.split('#')
    print (iaLine)
    ia = parts[0]
    interaction = 'UNKNOWN'
    if '0' in ia:
        interaction ="Coulomb"
    if '1' in ia:
        interaction ='HC'
    return (Ne,Nm, interaction)                                                                                                                                                                                           makPlot.py                                                                                          0000664 0001750 0001750 00000010307 13627704310 012047  0                                                                                                    ustar   chris                           chris                                                                                                                                                                                                                  #!/usr/bin/python3

import numpy as np
import sys, os
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


#fileName="HC-5-15-4000imps-vo-sigma-gap-var.dat"
fileName="bla.dat"

data=np.loadtxt(fileName)
print (data[:,1].shape)

data = data[~np.isnan(data).any(axis=1)]
new_array = [tuple(row) for row in data]
uniques = np.unique(new_array)

# Nimp = column 0 sigma = columm 1 vmax = colum 2 potvariance = colum3 gap = column4 spectrum --> rest
print (data.shape)
sigmas = data[:,1]
vomax = np.round(data[:,1]*data[:,2],7)
vmax = data[:,2]
allSigmas = np.unique(sigmas)
allVmax = np.unique(vmax)
allVmaxSigmas = np.unique(vomax)
print (allSigmas)
print (allVmaxSigmas)
resultList =[]
for vmaxSigma in allVmaxSigmas:
    subArray = data[np.where(data[:,1]*data[:,2] == vmaxSigma)]
    subArray = subArray[~np.isnan(subArray).any(axis=1)]
    print (subArray.shape)
    varMean = np.mean(subArray[:,3])
    varvar = np.var(subArray[:,3])**2
    gapMean = np.mean(subArray[:,4]-0.21)
    gapVariance = np.var(subArray[:,4])**2
    bandwidthMean = np.mean(subArray[:,7]-subArray[:,5])
    bandwidthVar = np.var(subArray[:,7]-subArray[:,5])*np.var(subArray[:,7]-subArray[:,5])
    
    print(vmaxSigma, gapMean , gapVariance,varMean,varvar)
    tuple = [vmaxSigma, gapMean , gapVariance,varMean,varvar,bandwidthMean, bandwidthVar]
    resultList.append(tuple)


# loop over whole array, use only those values which fit
resultArr = np.asarray(resultList)
resultArr = resultArr[~np.isnan(resultArr).any(axis=1)]
# FIt Data to polynomial
p4bandwidthw = np.poly1d(np.polyfit(resultArr[:,0], resultArr[:,5],6))
print ("Fit result Bandwidth")
print(p4bandwidthw)
fitx = np.linspace(np.amin(resultArr[:,0]),np.amax(resultArr[:,0]),15)
print (fitx)
fitybw = p4bandwidthw(fitx)
p4variance = np.poly1d(np.polyfit(resultArr[:,0], resultArr[:,3],6))
print ("Variance fit")
print (p4variance)
fityvariance = p4variance(fitx)

plt.plot(data[:,1]*data[:,2],data[:,4]-0.21,'ro')
plt.errorbar(resultArr[:,0],resultArr[:,1], yerr=resultArr[:,2], mfc='red',mec='green', ms = 20, mew=4, lw = 3 )
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('$V_{max}\sigma$')
plt.ylabel('$E_{gap}$ [enu]')
plt.show()
plt.errorbar(resultArr[:,0],resultArr[:,3], yerr=resultArr[:,4], mfc='red',mec='green', ms = 20, mew=4, lw = 3)
plt.plot(fitx,fityvariance, '--', lw = 3)
plt.xlabel('$V_{max}\sigma$')
plt.ylabel('$var(V(r))$')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.show()
plt.errorbar(resultArr[:,0],resultArr[:,5],yerr=resultArr[:,6],ms = 20, mew=4, lw = 3)
plt.plot(fitx,fitybw,'--', lw = 3)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('$V_{max}\sigma$')
plt.ylabel('Bandwidth')
plt.show()
plt.plot(data[:,3], data[:,4],'ro')
plt.xlabel('var(V(r))')
plt.ylabel('$E_{gap}$')
plt.show()
plt.plot(data[:,3],((data[:,8]-data[:,5])/(data[:,8]-data[:,7])-1), 'ro')
plt.xlabel('var(V(r))')
plt.ylabel('splitting')
plt.show()


exit()











sigmaList = ()
VmaxList = ()
VmaxSigmaList =()
for row in data:
    aSigma = row[1]
    if not aSigma in sigmaList:
        print ("neues Sigma"+str(aSigma))
        sigmaList = np.append(sigmaList,aSigma)
    avmax = row[2]
    if not avmax in VmaxList:
        VmaxList = np.append(VmaxList,avmax)
        print("neues Vmax" + str(avmax))
    vmaxSigma = np.round(aSigma*avmax,7)

    if not vmaxSigma in VmaxSigmaList:
        VmaxSigmaList = np.append(VmaxSigmaList,vmaxSigma)
        print (vmaxSigma)
print (VmaxSigmaList)
print (allVmaxSigmas)
resultArray=[]
for VmaxSigma in VmaxSigmaList:
    count = 0
    gapList =[]
    varianceList =[]
    for row in data:
        tempVsigma = np.round(row[1]*row[2],7)
        if VmaxSigma == tempVsigma:
            count =count +1
            gapList.append(row[4])
            varianceList.append(row[3])

    print (VmaxSigma,count)
    print (VmaxSigma, np.mean(np.asarray(gapList)) , np.var(np.asarray(gapList)))
plt.plot(data[:,1]*data[:,2],data[:,3],'ro')
plt.show()
                                                                                                                                                                                                                                                                                                                         NimpGradient_eval.py                                                                                0000664 0001750 0001750 00000002645 13633205760 014040  0                                                                                                    ustar   chris                           chris                                                                                                                                                                                                                  '''
Created on Feb 18, 2020

@author: chris
'''
import unittest
import Gaussian
import Spectrum
import numpy as np
import os
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


fileName = "HC5-15-Nimp-testData.dat"
data= np.loadtxt(fileName)                         
vosigmaArary = [1e-05*4000]
impArray = [100, 1000,2000,3000,4000,5000,6000,10000,20000,30000]
facArray=[0.95,1.0,1.05]
trialmax = 30

sigma = 0.05
dirCount = 0
allArray=[]
allgapList=[]
for nimpLoop in impArray:
    gapArray=[]
    for fac in facArray:
        nimp = int(nimpLoop*fac)
        subArray = data[np.where(data[:,0]==nimp)]
       
        gapAverage = np.mean(subArray[:,5])
        #print (nimp,gapAverage)
        gapArray.append((nimp,gapAverage))
        allArray.append((nimp,gapAverage))
        allgapList.append((nimp,gapAverage))
    gaps = np.array(gapArray)
    #print ("gaps ", gaps)
    diffgap = np.diff(gaps[:,1])
    diffImps = np.diff(gaps[:,0])
    print (nimpLoop,diffgap,diffImps, diffgap/diffImps)
    print (np.mean(np.abs(diffgap)/np.abs(diffImps)))

allgapArray = np.array(allgapList)
plt.plot(data[:,0],data[:,5], "ro")
plt.plot(allgapArray[:,0],allgapArray[:,1], "b+", markersize=20)
plt.show()
#plt.plot(

                                                                                           NimpGradient.py                                                                                     0000664 0001750 0001750 00000002607 13632421224 013021  0                                                                                                    ustar   chris                           chris                                                                                                                                                                                                                  '''
Created on Feb 18, 2020

@author: chris
'''
import unittest
import Gaussian
import Spectrum
import numpy as np
import os

def arrayToString(aArray):
    retVal = ""
    for val in aArray:
        retVal = retVal + " " + str(val)
    return retVal
                                
class Test(unittest.TestCase):


    def testName(self):
        vosigmaArary = [1e-05*4000]
        impArray = [100, 1000,2000,3000,4000]
        facArray=[0.95,1.0,1.05]
        trialmax = 30
        resFile = open('resultFileTest_NimpDiff.dat','a')
        sigma = 0.05
        dirCount = 0
        for nimpLoop in impArray:
            for fac in facArray:
                nimp = int(nimpLoop*fac)
                Vmax = 1e-05*4000 /(nimp*sigma)
                for index in range(0,trialmax):
                	x,y, pot, spectrum = Gaussian.createPotentialFromGaussian(dirCount,nimp,sigma,Vmax,'f')
                	potVar = np.var(pot)
                	dirCount = dirCount+1
                	gap = Spectrum.calculateGap(spectrum)
                	specString = arrayToString(spectrum)
                	os.chdir('..')
                	print (nimp, sigma,Vmax,potVar,gap,specString)
                	resFile.write(str(nimp)+" "+str(sigma)+ " "+str(Vmax)+" "+str(potVar)+" " +str(gap) + " " +specString+"\n")
        resFile.close()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
                                                                                                                         parse.py                                                                                            0000775 0001750 0001750 00000000212 13643570751 011557  0                                                                                                    ustar   chris                           chris                                                                                                                                                                                                                  #!/usr/bin/python3

import numpy as np

try:
    data = np.genfromtxt("test.dat", delimiter='')
except ValueError as e:
    print(e)
    
                                                                                                                                                                                                                                                                                                                                                                                      potentialAnalysisTest.py                                                                            0000664 0001750 0001750 00000002521 13630034751 015001  0                                                                                                    ustar   chris                           chris                                                                                                                                                                                                                  '''
Created on Feb 18, 2020

@author: chris
'''
import unittest
import Gaussian
import Spectrum
import numpy as np
import os

def arrayToString(aArray):
    retVal = ""
    for val in aArray:
        retVal = retVal + " " + str(val)
    return retVal
                                
class Test(unittest.TestCase):


    def testName(self):
        vosigmaArary = np.linspace(1e-06,8e-06, 8)
        trialmax = 30
        resFile = open('resultFileTest_coulomb.dat','a')
        sigmaArray = np.linspace(0.001,0.3,10)
        dirCount = 0
        for vosigma in vosigmaArary:
            for sigma in sigmaArray:
                Vmax = vosigma / sigma
                for index in range(0,trialmax):
                    x,y, pot, spectrum = Gaussian.createPotentialFromGaussian(dirCount,4000,sigma,Vmax,'f','Coulomb')
                    potVar = np.var(pot)
                    dirCount = dirCount+1
                    gap = Spectrum.calculateGap(spectrum)
                    specString = arrayToString(spectrum)
                    os.chdir('..')
                    print (4000, sigma,Vmax,potVar,gap,specString)
                    resFile.write("4000 "+str(sigma)+ " "+str(Vmax)+" "+str(potVar)+" " +str(gap) + " " +specString+"\n")
        resFile.close()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
                                                                                                                                                                               readResultFile.py                                                                                   0000775 0001750 0001750 00000001462 13624531524 013361  0                                                                                                    ustar   chris                           chris                                                                                                                                                                                                                  #!/usr/bin/python3

import numpy as np

resFile = open("testFile.dat",'r')
for line in resFile:


    content = line.split()
    # firstline has 13 entries, second has 4
    m_size = int(len(content))
    resArray = None
   
    print (content)
    if m_size > 10:
        imps = content[0]
        sigma = content[1]
        Vmax= content[2]
        potVar = content[3]
        gap = content[4]
        firstPart = True
        tempResayy = np.array((imps, sigma,Vmax,potVar,gap))
        print (tempResayy)
        if not resArray:
            print ("Create resArray")
            resArray = tempResayy
        else:
            resArray = np.concatenate((resArray,tempResayy))
        print (resArray)

    else:
        secondPart = True
#    print (imps, sigma,Vmax,potVar,gap)

print ("Array")
print (resArray)
                                                                                                                                                                                                              reformatData.py                                                                                     0000664 0001750 0001750 00000002341 13627423244 013054  0                                                                                                    ustar   chris                           chris                                                                                                                                                                                                                  import numpy as np
import sys, os

fileName = "bla.dat"

file = open(fileName,"r")
lines = file.readlines()
for line in lines:
    if  line.strip():
        tempLine = line.strip()
        tempLine = tempLine.strip("]")
        content = tempLine.split(" ")
    
   #     last_entry = content[-1]
        last_entry = content.pop()
        #print (last_entry)
        lastEntryContent = last_entry.split('.')
        # length of first entry
        #print (lastEntryContent)
        decNumbers = len(lastEntryContent[0])
        ersteVorkomma = lastEntryContent[0];
        
        #print (decNumbers)
        # last decNumberEntrsy from second
        nachkommaletzteZahl = lastEntryContent.pop()
        #print(nachkommaletzteZahl)
        try:
            mittlereZahl  = lastEntryContent.pop()
         #   print (mittlereZahl)
        except  IndexError:
            continue
        dezErsteZahl = mittlereZahl[:-2]
        #print (dezErsteZahl)
        ersteZahl = ersteVorkomma+"."+dezErsteZahl
        zweiteDez = mittlereZahl[-2:]
       # print (ersteZahl)
        zweiteZahl = zweiteDez+"."+nachkommaletzteZahl
        print (*content, ersteZahl, zweiteZahl)
        
    else:
        print ("Empty")
        print (line)
file.close()
                                                                                                                                                                                                                                                                                               reformatFile.py                                                                                     0000664 0001750 0001750 00000001126 13643624732 013065  0                                                                                                    ustar   chris                           chris                                                                                                                                                                                                                  '''
Created on Apr 9, 2020

@author: chris
'''

if __name__ == '__main__':
    pass

import os, sys
if len(sys.argv)<2:
    fileName = "resultTest.dat"
else:
    fileName = sys.argv[1]


longlineRead = False
file = open(fileName,"r")
Lines = file.readlines()
file.close()
curLine = 0
buffer = ""
for line in Lines:
    if '#' in line:
     #   print (line)
        continue

    if not line or "" == line:
        continue
    longlineRead = len(line.split()) > 2
    
    #print (longlineRead)
    if  longlineRead:
        buffer = line.strip()
    else:
        print (buffer + " "+line.strip())                                                                                                                                                                                                                                                                                                                                                                                                                                          ResultDirectory.py                                                                                  0000664 0001750 0001750 00000011142 13642654436 013613  0                                                                                                    ustar   chris                           chris                                                                                                                                                                                                                  '''
Created on Feb 16, 2020

@author: chris
'''
import os, sys
import numpy as np
import Gaussian
import CorrelationFit
from symbol import except_clause
class resultDirectory(object):
    '''
    classdocs
    '''
    Ne = 0
    def __init__(self, dirName):
        '''
        Constructor
        '''
        self.potFile = ""
        self.spectrumFile =""
        self.evCorrelationFile =""
        self.vvCorrelationFile =""
        self.randomFile = ""
        self.gaussianFile=""
        self.Ne = 0
        self.Nm = 0
        self.interaction ="Unknown"
        self.home = dirName
        self.sigmax = "Unknown"
        self.Vmax = "Unknown"
        self.impCount = "Unknown"
        self.sigmay = "Unknown"
    
        for aFile in os.listdir(dirName):
            print (aFile)
            if "PotentialArray" in aFile:
                self.potFile = aFile
                continue
            if "spectrum" in aFile:
                self.spectrumFile = aFile
                continue
            if "evCorrelation_" in aFile and not "temp" in aFile:
                # ignore temp file
                self.evCorrelationFile = aFile
                continue
            if "vortexVortexCorrelation_" in aFile and not "temp" in aFile:
                self.vvCorrelationFile = aFile
                continue
            if "gaussian.par" in aFile:
                self.gaussianFile = aFile
                continue
            if "random.dat" in aFile:
                self.randomFile = aFile
                continue
        print ("Done reading")
        os.chdir(dirName)
        self.evaluate()

    
    
    def print(self): 
        print("============================== Content of " + self.home+" ===============")
        print (self.evCorrelationFile, self.vvCorrelationFile, self.potFile, self.gaussianFile, self.randomFile,self.spectrumFile)
        print ("============================== EnDE ===========================")
        
        
    def evaluate(self):
        # get ne, Nm
        # getSigma
        try:
            if self.randomFile:
                try:
                    self.impCount, self.Vmax, self.sigmax, self.sigmay = Gaussian.readRandom(self.randomFile)
                except:
                    print("Reading failed")
                    
                print (self.impCount, self.sigmax, self.sigmay, self.Vmax)
    
            # def get lcorr and sigmax,sigmay
            if self.potFile:
                self.lcorr = Gaussian.calculateAutoCorrelationFromFile(self.potFile)
                self.results2d = Gaussian.calculateAutoCorrelation2dFromFile(self.potFile)
                self.results2d = self.results2d[:2]
            
                print (self.lcorr, self.results2d)
            else:
                self.lcorr = "UNKNOWN"
                self.results2d="UNKNOWN"
            # getSpectrum
            if self.spectrumFile:
                self.spectrum =Gaussian.readSpectrumFile(self.spectrumFile)
                print(self.spectrum)
            else:
                self.spectrum = 'UNKNOWN'
            if self.evCorrelationFile:
                try:
                    self.evMax = CorrelationFit.fitAndPlot2(self.evCorrelationFile,True,plot = False)
                except:
                    self.evMax= "UNKNOWN"
            else:
                self.evMax= "UNKNOWN"
            if self.vvCorrelationFile:
                try:
                    self.vvMax = CorrelationFit.fitAndPlot2(self.vvCorrelationFile,True,plot = False)
                except:
                    self.vvMax= "UNKNOWN"
            else:
                self.vvMax= "UNKNOWN"
            
            if self.gaussianFile:
                (self.Ne, self.Nm, self.interaction) = Gaussian.readGaussianPar(self.gaussianFile)  
        except:
            self.lcorr = "UNKNOWN"
            self.results2d="UNKNOWN"    
            self.spectrum = 'UNKNOWN'
            self.evMax= "UNKNOWN"
            self.vvMax= "UNKNOWN"
    
    def electronNumber(self):
        return (self.Ne)
            
            
    def printValues(self, resultFile):
        print (self.Ne, self.Nm, self.interaction, self.impCount, self.Vmax, self.sigmax, self.sigmay,self.lcorr, np.ravel(self.results2d),  self.evMax[0], self.vvMax[0],  self.spectrum)
        resultFile.write(str(self.Ne) + " ," +str(self.Nm)+ " ," + str(self.interaction))
        resultFile.write(str(self.impCount) + " ," + str(self.Vmax)+ " ," + str(self.sigmax) + " ," + str(self.sigmay) + " ," + str(self.lcorr)+ " ,")
        resultFile.write( str(np.ravel(self.results2d))+ " ," +  str(self.evMax[0])+ " ," + str(self.vvMax[0])+ " ," +  str(self.spectrum)+ "\n")
    
                                                                                                                                                                                                                                                                                                                                                                                                                                              resultdirectorytest.py                                                                              0000664 0001750 0001750 00000003364 13643617112 014611  0                                                                                                    ustar   chris                           chris                                                                                                                                                                                                                  '''
Created on Feb 16, 2020

@author: chris
'''
import unittest
import os
import ResultDirectory as rd

import CorrelationFit
from DirectoryResult import ParseDirectory

class Test(unittest.TestCase):
    

    def testParseDir(self):
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
        
        

    def testParseDir2(self):
        ad = ParseDirectory.ParseDirectory("/home/chris/eclipse-workspace/tkiterTrial/cli/test/dir_2")
        self.assertEqual(ad.ne,'5',"")
        self.assertEqual(ad.Nm,'13',"" )
        self.assertEqual(ad.interaction,"Coulomb")
        file = open("./EinAnderesFile.dat","w")
        ad.printValues(file)

    def atestName(self):
        directoryName = "testDirectory/test/dir_1"
        aDir = rd.resultDirectory(directoryName)
        self.assertEqual(6, rd.electronNumber(), "Msg")
        aDir.printValues()
       

    def atestFit(self):
        fileName = "test/dir_1/evCorrelation_1.dat"
        
        results= CorrelationFit.fitAndPlot2(fileName,True,plot = False)
        print (results)
        max = results[0]
        print (max)
    def atestDirectory(self):
        directoryName="/home/chris/cluster/clusterResults2/loopsParallel/coulomb"
        aDir = rd.resultDirectory(directoryName)
        resFile = open("Results.dat")
        aDir.printValues(resFile)
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()                                                                                                                                                                                                                                                                            Spectrum.py                                                                                         0000664 0001750 0001750 00000000530 13623014424 012233  0                                                                                                    ustar   chris                           chris                                                                                                                                                                                                                  '''
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
    return bw                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
