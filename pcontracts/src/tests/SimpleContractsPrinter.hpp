/**
 * \internal
 * File:  SimpleContractsPrinter.hpp
 * \endinternal
 *
 * @file
 * @brief
 * Contracts visitor for printing PAUL CONTRACT annotations.
 *
 * @details
 * Simple contracts visitor class that looks for and prints CONTRACT 
 * annotations.
 *
 * @htmlinclude copyright.html
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
