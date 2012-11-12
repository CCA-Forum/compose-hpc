/*
 * File:  test_PaulContractsDecorator.cpp
 *
 *
 * @file
 * @section DESCRIPTION
 * Test driver for the PaulContractsDecorator class.
 *
 *
 * @section SOURCE
 * Based on libpaul's test_PaulDecorate.cpp.
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
}
