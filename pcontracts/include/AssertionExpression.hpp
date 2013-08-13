/**
 * \internal
 * File:           AssertionExpression.hpp
 * Author:         T. Dahlgren
 * Created:        2012 November 9
 * Last Modified:  2013 August 2
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

/**
 * Assertion expression support states.
 */
typedef enum AssertionSupport__enum {
  /** ADVISORY:  Advisory only. */
  AssertionSupport_ADVISORY,
  /** COMMENT:  A _brief_ comment=>Hack to re-use infrastructure for STATS. */
  AssertionSupport_COMMENT,
  /** EXECUTABLE:  Executable in C (currently as-is). */
  AssertionSupport_EXECUTABLE,
  /** FILENAME:  A filename =>Hack to re-use infrastructure for INIT filename.*/
  AssertionSupport_FILENAME,
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
    AssertionExpression(std::string l, std::string expr, 
      AssertionSupportEnum level) : d_label(l), d_expr(expr), d_level(level) {}

    /** Destructor */
    ~AssertionExpression() {}

    /** Return the optional label. */
    std::string label() { return d_label; }

    /** Return the expression. */
    std::string expr() { return d_expr; }

    /** Return the level of support for the expression. */
    AssertionSupportEnum support() { return d_level; }

    /** 
     * Return a string representation of the expression.  
     *
     * @param[in] sep  Field separator.
     * @return         Field-separated string representation.
     */
    std::string str(std::string sep) 
    { 
      std::ostringstream rep;
      rep << d_label << sep << d_expr << sep;
      switch (d_level)
      {
      case AssertionSupport_ADVISORY:
        rep << "Advisory";
        break;
      case AssertionSupport_COMMENT:
        rep << "Comment";
        break;
      case AssertionSupport_EXECUTABLE:
        rep << "Executable";
        break;
      case AssertionSupport_FILENAME:
        rep << "Filename";
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
    std::string          d_label;

    /** The assertion expression (text). */
    std::string          d_expr;

    /** The level of translation support associated with the expression. */
    AssertionSupportEnum d_level;
};  /* class AssertionExpression */

#endif /* include_Assertion_Expression_hpp */
