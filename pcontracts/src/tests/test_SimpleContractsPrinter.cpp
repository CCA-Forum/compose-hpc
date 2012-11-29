/**
 * \internal
 * File:  test_SimpleContractsPrinter.cpp
 * \endinternal
 *
 * @file
 * @brief
 * Test driver for the SimpleContractsPrinter class.
 *
 * @htmlinclude copyright.html
 */
#include "PaulContractsCommon.h"
#include "PaulContractsDecorator.hpp"
#include "rose.h"
#include "SimpleContractsPrinter.hpp"

using namespace std;

int 
main(int argc, char * argv[])
{
  //Build the ROSE AST
  SgProject* sageProject  = frontend (argc , argv) ;
  ROSE_ASSERT (sageProject != NULL);

  // Decorate the AST with the Contract annotations
  paulContractsDecorate(sageProject, "../conf/Contracts.paulconf");
  
  // Run internal consistency tests
  AstTests::runAllTests(sageProject);

  // Process 
  SimpleContractsPrinter scp = new SimpleContractsPrinter();
  scp.traverseInputFiles(sageProject, preorder);

  delete scp;
  return 0;
}  /* main */
