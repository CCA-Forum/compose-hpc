/**
 * \internal
 * File:          ContractAssertionPrinter.hpp
 * Author:        T. Dahlgren
 * Created:       2012 July 6
 * Last Modified: 2012 November 28
 * \endinternal
 *
 * @file
 * @brief 
 * Class for printing contract clause assertions detected in ROSE AST comments.
 *
 * @details
 * Simple class implementation for illustrating the use of basic ROSE features 
 * and a rudimentary contract clause parser for extracting and printing 
 * contract clause assertions.  
 *
 * @htmlinclude copyright.html
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
     * @param[in] node  Current AST node.
     */
    void virtual visit(SgNode* node);

}; /* ContractAssertionPrinter */

#endif /* include_Contract_Assertion_Printer_hpp */
