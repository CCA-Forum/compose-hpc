/**
 * File:          ContractPrinter.hpp
 * Author:        T. Dahlgren
 * Created:       2012 July 6
 * Last Modified: 2012 August 3
 *
 * @section DESCRIPTION
 * Simple class used for illustrating the use of basic ROSE features for 
 * printing contract clause comments.  It is NOT intended for contract
 * enforcement use.
 *
 * @section LICENSE
 * TBD
 */

#ifndef include_Contract_Printer_hpp
#define include_Contract_Printer_hpp

#include "rose.h"


class ContractPrinter : public AstSimpleProcessing
{
  public:
    /**
     * Simple constructor.
     */
    ContractPrinter() {};

    /**
     * Process the current AST node passed by the front end.  This
     * simply involves printing contract clause comments.
     *
     * @param node  Current AST node.
     */
    void virtual visit(SgNode* node);
}; /* ContractPrinter */

#endif /* include_Contract_Printer_hpp */
