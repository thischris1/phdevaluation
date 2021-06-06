#!/usr/bin/python3

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
