/*
 * File:         test_InvariantAnnotation.cpp
 * Description:  Test driver to check a basic invariant annotation.
 * Source:       Based on libpaul's test_SXAnnotation.cpp
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
