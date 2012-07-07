/**
 * File:          ContractPrinter.hpp
 * Author:        T. Dahlgren
 * Created:       2012 July 6
 * Last Modified: 2012 July 6
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
  protected:
    /**
     * Determines if the specified directive type is a comment.
     *
     * @param dType  The type of preprocessing directive.
     * @return       true if the directive is a C/C++ comment; otherwise, false.
     */
    bool isComment(PreprocessingInfo::DirectiveType dType);

    /**
     * Prints the specified comment followed by the node type, line, and file 
     * information of the associated node.
     *
     * @param node  The associated AST node.
     * @param cmt   The comment to be printed.
     * @return       true if the directive is a C/C++ comment; otherwise, false.
     */
    void printLineComment(SgNode* node, const char* cmt);


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
