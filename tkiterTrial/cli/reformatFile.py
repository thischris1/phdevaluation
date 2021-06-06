'''
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
        print (buffer + " "+line.strip())