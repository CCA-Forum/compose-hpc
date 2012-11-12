/*
 * File:  SimpleContractsPrinter.cpp
 *
 *
 * @file
 * @section DESCRIPTION
 * Simple contracts visitor class that looks for and prints CONTRACT 
 * annotations.
 *
 *
 * @section SOURCE
 * Based on PAUL's example_traversal.cpp.
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
//#include "PaulDecorate.h"
//#include "KVAnnotationValue.h"
#include "PaulContractsCommon.h"
#include "rose.h"
#include "SimpleContractsPrinter.hpp"

/**
 * Visit the node to determine if there are any contract annotations.
 * If so, then print them out.
 */
void 
SimpleContractsPrinter::visit(SgNode *node) 
{
  Annotation *annot = (Annotation *)node->getAttribute(L_CONTRACTS_TAG);

  if (annot != NULL) {
    // 1. Retrieve value
    //KVAnnotationValue *val = (KVAnnotationValue *)annot->getValue();
    // 2. Make sure it is a KV annotation value
    //val = isKVAnnotationValue(val);
    //ROSE_ASSERT(val != NULL);

    // 1. Retrieve the contract clause information
    // 2. make sure it is a contract clause
    std::cout << "Found annotated node:" << node->class_name() << std::endl;
    //val->print();
    std::cout << "\n";
  }
} /* SimpleContractsPrinter::visit */
