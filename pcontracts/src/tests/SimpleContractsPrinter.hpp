/*
 * File:         SimpleContractsPrinter.hpp
 * Description:  Simple contracts visitor class that looks for and prints
 *               CONTRACT annotations.
 * Source:       Based on paul's example_traversal.cpp
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
