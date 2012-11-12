/*
 * File:  SimpleContractsPrinter.hpp
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
#ifndef included_SimpleContractsPrinter_hpp
#define included_SimpleContractsPrinter_hpp

#include "rose.h"
#include "PaulContractsCommon.h"

/**
 * Class:   SimpleContractsPrinter
 */ 
class SimpleContractsPrinter : public AstSimpleProcessing 
{
  protected:
    void virtual visit(SgNode *node);
  
  public:
    /* Default constructor */
    SimpleContractsPrinter() {}
  
    /* Destructor */
    virtual ~SimpleContractsPrinter() {}
}; /* end class SimpleContractsPrinter */

#endif /* included_SimpleContractsPrinter_hpp */
