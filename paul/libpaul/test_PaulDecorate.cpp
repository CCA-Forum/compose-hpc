
#include "PaulDecorate.h"

using namespace std;


int main( int argc, char * argv[] )
{
  //Build the AST used by ROSE
  SgProject* sageProject  = frontend (argc , argv) ;
  ROSE_ASSERT (sageProject != NULL);

  // decorate the AST with the PAUL annotations
  paulDecorate (sageProject, "example.paulconf");

 // Run internal consistency tests on AST
  AstTests::runAllTests(sageProject);

  // Generate source code from AST and call the vendor's compiler
  return backend(sageProject);
}
