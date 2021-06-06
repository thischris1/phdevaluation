import numpy as np
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
