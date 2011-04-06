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

int main(int argc, char **argv) {
  SgProject *project = frontend(argc,argv);
  AstTests::runAllTests(project);
  return backend(project);
}
