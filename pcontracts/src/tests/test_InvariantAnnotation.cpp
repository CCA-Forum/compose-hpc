/*
 * File:         test_InvariantAnnotation.cpp
 *
 *
 * @file
 * @section DESCRIPTION
 * Test driver to check a basic invariant annotation.
 *
 *
 * @section SOURCE
 * Based on libpaul's test_SXAnnotation.cpp.
 *
 *
 * @section COPYRIGHT
 * Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Tamara Dahlgren <dahlgren1@llnl.gov>.
 * 
 * LLNL-CODE-473891.
 * All rights reserved.
 * 
 * This software is part of COMPOSE-HPC. See http://compose-hpc.sourceforge.net/
 * for details.  Please read the COPYRIGHT file for Our Notice and for the 
 * BSD License.
 */
#include <iostream>
//#include <stdio.h>
#include "rose.h"
#include "InvariantAnnotation.h"

int 
main( int argc, char * argv[] )
{
  // ToDo/TBD:  Need to modify the following to match the actual constructor's
  // arguments.
  string iaStr1 = "all_pos_weights: onlyPosWeights()";
  InvariantAnnotation inv (iaStr);
  inv.print();
  cout << "\n";

  // ToDo/TBD:  What is the significance of tag to Annotation?
  Annotation a = Annotation(inv, (SgLocatedNode *)(NULL), "tag", &a);
  // ToDo/TBD: Need to modify the output according to the actual invariant
  //   contents
  cout << "type= " << a.getTag() << endl;
  cout << "str= " << a.getValueString() << endl;
  InvariantAnnotation *ia = isInvariantAnnotation(a.getTag());
  ia->print();

  cout << "Finished\n";
  return 0;
}
