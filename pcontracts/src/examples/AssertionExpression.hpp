/**
 * \internal
 * File:           AssertionExpression.hpp
 * Author:         T. Dahlgren
 * Created:        2012 November 9
 * Last Modified:  2013 January 31
 * \endinternal
 *
 * @file
 * @brief
 * Basic structures for managing assertion expression data.
 *
 * @htmlinclude copyright.html
 */

#ifndef include_Assertion_Expression_hpp
#define include_Assertion_Expression_hpp

#include <string>
#include <sstream>

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

    /** 
     * Return a string representation of the expression.  
     *
     * @param[in] sep  Field separator.
     * @return         Field-separated string representation.
     */
    string str(string sep) 
    { 
      ostringstream rep;
      rep << d_label << sep << d_expr << sep;
      rep << (d_isFirst ? "First" : "Not First") << sep;
      switch (d_level)
      {
      case AssertionSupport_ADVISORY:
        rep << "Advisory";
        break;
      case AssertionSupport_EXECUTABLE:
        rep << "Executable";
        break;
      case AssertionSupport_UNSUPPORTED:
        rep << "Unsupported";
        break;
      default:
        rep << "UNKNOWN";
        break;
      }
      return rep.str(); 
    }

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
