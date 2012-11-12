/**
 * File:          ContractAssertionPrinter.hpp
 * Author:        T. Dahlgren
 * Created:       2012 July 6
 * Last Modified: 2012 November 12
 *
 *
 * @section DESCRIPTION
 * Simple class used for illustrating the use of basic ROSE features for 
 * printing contract clause assertions.  It is NOT intended for contract
 * enforcement use.
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

#ifndef include_Contract_Assertion_Printer_hpp
#define include_Contract_Assertion_Printer_hpp

#include "rose.h"
#include "ContractPrinter.hpp"


class ContractAssertionPrinter : public ContractPrinter
{
  public:
    /**
     * Simple constructor.
     */
    ContractAssertionPrinter() {};

    /**
     * Process the current AST node passed by the front end, identifying
     * and printing individual contract clause assertions.
     *
     * @param node  Current AST node.
     */
    void virtual visit(SgNode* node);

}; /* ContractAssertionPrinter */

#endif /* include_Contract_Assertion_Printer_hpp */
