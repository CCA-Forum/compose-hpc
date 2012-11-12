/*
 * File:  test_SimpleContractsPrinter.cpp
 *
 *
 * @file
 * @section DESCRIPTION
 * Test driver for the SimpleContractsPrinter class.
 *
 *
 * @section SOURCE
 * Based on paul's example_traversal.cpp.
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
#include "PaulContractsCommon.h"
#include "PaulContractsDecorator.hpp"
#include "rose.h"
#include "SimpleContractsPrinter.hpp"

using namespace std;

int 
main( int argc, char * argv[] )
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
}
