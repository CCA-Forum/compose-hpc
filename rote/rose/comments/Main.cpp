// This program should perform an identity source-to-source translation to 
// the input program.
//
// To build:
// $ g++ -I$BOOST_HOME/include -I$ROSE_HOME/include -L$BOOST_HOME/lib 
//       -L$ROSE_HOME/lib -lrose identity.cpp -o identity
//
// To run:
// $ ./identity input.cpp
//

#include "rose.h"
#include "TestVisitor.h"

int main(int argc, char **argv) {
  SgProject *node = frontend(argc,argv);
  TestVisitor v;
  v.traverseInputFiles(node,preorder);
  return 0;
}
