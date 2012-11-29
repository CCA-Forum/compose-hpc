/**
 * \internal
 * File:  test_PaulContractsDecorator.cpp
 * \endinternal
 *
 * @file
 * @brief
 * Test driver for the PaulContractsDecorator class.
 *
 * @htmlinclude copyright.html
 */
#include "PaulContractsDecorator.hpp"

int 
main( int argc, char * argv[] )
{
  //Build the ROSE AST
  SgProject* sageProject  = frontend (argc , argv) ;
  ROSE_ASSERT (sageProject != NULL);

  // Decorate the AST with Contract annotations
  paulContractsDecorate(sageProject, "../conf/Contracts.paulconf");

 // Run internal consistency tests
  AstTests::runAllTests(sageProject);

  // Generate source code from AST and call the vendor's compiler
  return backend(sageProject);
}  /* main */
