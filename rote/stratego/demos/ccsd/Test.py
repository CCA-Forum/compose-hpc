#!/usr/bin/env python

import sys
import os
import stat
from os.path import basename

testname=str(sys.argv[1])
executable=str(sys.argv[2])
testcase=str(sys.argv[3])
resultsDir=str(sys.argv[4])

os.chdir(resultsDir)
os.system("%s -c %s" %(executable,testcase))
strategoScriptPath=resultsDir+"/"+testname+"_ccsd.str"
strategoScriptName=resultsDir+"/"+testname+"_ccsd"

if not os.path.exists(strategoScriptPath):
    print "Unable to generate Stratego rewrite rules for test : %s" %(testcase)
    sys.exit(1)
  
os.system("strc -i %s -la stratego-lib" %(strategoScriptPath))
os.system("src2term --stratego -i %s -o input.term" %(testcase))
os.system("%s -i input.term > output.term" %(strategoScriptName))
os.system("term2src --stratego output.term")
os.system("rm %s *.c *.lo *.dep *.o *.f90.dot *.term" %(strategoScriptName))

transformedCode = resultsDir+"/"+testname+".unparsed.f90"
      
if not os.path.exists(transformedCode):
    print "Unable to generate transformed code for test : %s" %(testcase)
    sys.exit(1)
      
os.rename(transformedCode, resultsDir+"/"+testname+"_transformed.f90")
