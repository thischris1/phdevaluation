'''
Created on Jan 18, 2021

@author: chris
'''
import os,glob
import sys
import shutil
import pathlib
if __name__ == '__main__':
    pass
targetDir="/home/chris/cluster/clusterResults2/6_18/hardcore/prepare"

if not os.path.exists(targetDir):
    sys.exit()
os.chdir(targetDir)
# run over all impurites_ files
impFiles = glob.glob('dir_*')
count = 0
for aFile in impFiles:
    print (count, aFile)
    dirName = targetDir+ "/dir_"+str(count)
    os.chdir(dirName)
    os.system("~/eclipse-workspace/gaussian/Release/gaussian -p")
    count = count +1
    os.chdir('..')
    
    
    