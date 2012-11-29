/**
 * \internal
 * File:          ContractPrinter.hpp
 * Author:        T. Dahlgren
 * Created:       2012 July 6
 * Last Modified: 2012 November 28
 * \endinternal
 *
 * @file
 * @brief
 * Class for printing contract clause comments from the ROSE AST.
 *
 * @details
 * Simple class implementation for illustrating the use of basic ROSE features 
 * for identifying and printing contract clause comments.  
 *
 * @htmlinclude copyright.html
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
     * @param[in] node Current AST node.
     */
    void virtual visit(SgNode* node);
}; /* ContractPrinter */

#endif /* include_Contract_Printer_hpp */
