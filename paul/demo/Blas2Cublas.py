'''
Created on Jul 6, 2011

@author: Ajay Panyala
'''

#!/usr/bin/env python

import os
import sys

if len(sys.argv) <= 2: 
   print "\nUsage: ./Blas2Cublas.py path-to-file BLAS_OPTION autoAnnot"
   print "BLAS_OPTION is one of {BLAS,MKL}\n"
   sys.exit()

filePath = sys.argv[1]

autoAnnotOption = ""

try:
    autoAnnotOption = sys.argv[3]
except:
    print ""

sindex = 0
try:
    sindex = filePath.rindex("/")+1
except ValueError:
    print ""
    
fileName = filePath[sindex:]
filePath = filePath[0:sindex]

if filePath: os.chdir(filePath)

' Read the linker file. '
paulHome = os.environ.get("PAUL_HOME")
paul = paulHome+"/paul"
linkerPath = paulHome+"/blas2cublas/linker"
simpleTrans = paulHome+"/blas2cublas/SimpleTranslator"
autoAnnot = paulHome+"/blas2cublas/addAnnot.cocci"

linkFile = open(linkerPath, mode='r')

link = ""

for line in linkFile:
    if line.startswith('%'): link += line[line.index("%")+1:].strip()
        
linkFile.close

inc = sys.argv[2]
if(inc == "MKL"):
    linker = "-I$MKL_HOME/include %s" %(link)
elif(inc == "BLAS"): linker = "-I$BLAS_HOME/include %s" %(link)
else:
    print "Invalid blas library options. Valid options are BLAS, MKL."
    exit(1)
    
fName = fileName[0:fileName.rindex(".")]
extension = fileName[fileName.rindex(".")+1:]

cocciOptions = ""
if(extension != "c"): cocciOptions = "-c++"

os.system("%s -rose:skipfinalCompileStep %s %s" %(simpleTrans,fileName,linker))

if(len(autoAnnotOption) != 0):
    os.system("spatch %s -cocci_file %s rose_%s -o rose_trans_%s > /dev/null" %(cocciOptions,autoAnnot,fileName,fileName))
    os.system("sed -i 's/%s/%s/g' rose_trans_%s" %("*\/;","*\/",fileName))
else:
    os.system("cp rose_%s rose_trans_%s" %(fileName,fileName))

print "Running PAUL"
os.system("%s rose_trans_%s.%s %s" %(paul,fName,extension,linker))

blasCocci = "rose_trans_%s_blasCalls.cocci" %(fName)

if(os.path.exists(blasCocci)):
    print "Transforming rose_trans_%s.%s" %(fName,extension)

    os.system("spatch %s -cocci_file  %s rose_trans_%s.%s\
              -o rose_trans_%s_cublas.cu > /dev/null" %(cocciOptions,blasCocci,fName,extension,fName))

    if(len(link) != 0):
	    print "\nCompiling rose_trans_%s_cublas.cu ..." %(fName)
	    compileCuda = "nvcc rose_trans_%s_cublas.cu -o rose_trans_%s_cublas -lcuda -lcudart -lcublas %s" %(fName,fName,linker)
	    print compileCuda
	    os.system(compileCuda)

#print "Building both versions of rose_trans_%s" %(fName)
#os.system(icc ..)


else:
    print "File "+blasCocci+" not generated by PAUL.\n \
           Blas2Cublas Transformation cannot be applied to " + fileName
    
    
