/**
 * File:           AssertionExpression.hpp
 * Author:         T. Dahlgren
 * Created:        2012 November 9
 * Last Modified:  2012 November 9
 *
 * @file
 * @section DESCRIPTION
 * Basic assertion expression data.
 *
 * @section SOURCE
 * The class was originally part of the initial ContractInstrumenter.cpp,
 * then ContractsProcessor.hpp.  It was separated, along with ContractComment,
 * and the supporting enumeration added (from contractClauseTypes.hpp) to
 * improve modularity.
 * 
 * @section LICENSE
 * TBD
 */

#ifndef include_Assertion_Expression_hpp
#define include_Assertion_Expression_hpp

using namespace std;

/**
 * Assertion expression support states.
 */
typedef enum AssertionSupport__enum {
  /** ADVISORY:  Advisory only. */
  AssertionSupport_ADVISORY,
  /** EXECUTABLE:  Executable in C (currently as-is). */
  AssertionSupport_EXECUTABLE,
  /** UNSUPPORTED:  Known to include an unsupported annotation. */
  AssertionSupport_UNSUPPORTED
} AssertionSupportEnum;


/**
 * Class for managing the contents of a PAUL contract assertion expression.
 *
 * This class manages the information extracted from an assertion expression
 * within a PAUL contract structured comment.  Consequently, it corresponds
 * to one of the assertions -- executable or advisory -- withinn the contract
 * clause.
 */
class AssertionExpression 
{
  public:
    /** Constructor */
    AssertionExpression(string l, string expr, AssertionSupportEnum level,
      bool isFirst) 
      : d_label(l), d_expr(expr), d_level(level), d_isFirst(isFirst) {}

    /** Destructor */
    ~AssertionExpression() {}

    /** Return the optional label. */
    string label() { return d_label; }

    /** Return the expression. */
    string expr() { return d_expr; }

    /** Return the level of support for the expression. */
    AssertionSupportEnum support() { return d_level; }

    /** 
     * Return true if the assertion is the first in the clause; false, 
     * otherwise. 
     */
    bool isFirst() { return d_isFirst; }

  private:
    /** The optional label associated with the expression. */
    string               d_label;

    /** The assertion expression (text). */
    string               d_expr;

    /** The level of translation support associated with the expression. */
    AssertionSupportEnum d_level;

    /** Indicates whether the expression is the first in the clause. */
    bool                 d_isFirst;
};  /* class AssertionExpression */

#endif /* include_Assertion_Expression_hpp */
