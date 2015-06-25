/**
 * \internal
 * File:           ContractComment.hpp
 * Author:         T. Dahlgren
 * Created:        2012 November 9
 * Last Modified:  2015 June 11
 * \endinternal
 *
 * @file
 * @brief
 * Basic structures for managing contract comment/clause data.
 *
 * @htmlinclude copyright.html
 */

#ifndef include_Contract_Comment_hpp
#define include_Contract_Comment_hpp

#include <list>
#include <sstream>
#include "rose.h"
//#include "contractOptions.h"
#include "AssertionExpression.hpp"

#define PPIDirectiveType PreprocessingInfo::DirectiveType


/**
 * Supported structured contract comment types.
 */
typedef enum ContractComment__enum {
  /** NONE:  No contract comment present. */
  ContractComment_NONE,
  /** INVARIANT:  An invariant clause comment. */
  ContractComment_INVARIANT,
  /** PRECONDITION:  A precondition clause comment. */
  ContractComment_PRECONDITION,
  /** POSTCONDITION:  A postcondition clause comment. */
  ContractComment_POSTCONDITION,
  /** ASSERT:  An assertion clause comment. */
  ContractComment_ASSERT,
  /** INIT:  An initialization comment. */
  ContractComment_INIT,
  /** FINAL:  A finalization comment. */
  ContractComment_FINAL,
  /** STATS:  A 'dump enforcement statistics' comment. */
  ContractComment_STATS
} ContractCommentEnum;


/**
 * Mapping of contract comment to contract clauses.  MUST be kept in sync
 * with ContractCommentEnum and reflect corresponding ContractClauseEnum
 * entries.
 */
static const ContractClauseEnum ContractCommentClause[] = {
  /** NONE:  No corresponding contract clause. */
  ContractClause_NONE,
  /** INVARIANT:  Invariant contract clause. */
  ContractClause_INVARIANT,
  /** PRECONDITION:  Precondition contract clause. */
  ContractClause_PRECONDITION,
  /** POSTCONDITION:  Postcondition contract clause. */
  ContractClause_POSTCONDITION,
  /** ASSERT:  Assertion contract clause. */
  ContractClause_ASSERT,
  /** INIT:  No corresponding contract clause. */
  ContractClause_NONE,
  /** FINAL:  No corresponding contract clause. */
  ContractClause_NONE,
  /** STATS:  No corresponding contract clause. */
  ContractClause_NONE
};


/**
 * Class for managing the contents of a PAUL contract structured comment.
 *
 * This class manages the information extracted from a PAUL contract
 * structured comment.  Consequently, it generally corresponds to a 
 * contract clause, which consists of a list of assertions; however, not
 * all comments contain assertions.
 *
 * @todo  Consider (some day) refactoring to recognize the distinction 
 *   between the types of contract comments (ie, those with and those
 *   without assertions).
 */
class ContractComment
{
  public:
    /** Constructor */
    ContractComment(ContractCommentEnum t, PPIDirectiveType dt): 
      d_type(t), d_dirType(dt), d_numExec(0), d_needsResult(false), 
      d_isInInit(false), d_isPure(false) {}

    /** Destructor */
    ~ContractComment() { d_aeList.clear(); }

    /** Return the contract comment type. */
    ContractCommentEnum type() { return d_type; }

    /** Return the contract clause type. */
    ContractClauseEnum clause() { return ContractCommentClause[d_type]; }

    /** 
     * Add the assertion expression to the contract clause. 
     *
     * @param[in] ae  Current assertion expression
     */
    void add(AssertionExpression ae) 
    { 
      d_aeList.push_front(ae); 
      if (ae.support() == AssertionSupport_EXECUTABLE) { d_numExec += 1; }
      if (ae.hasResult()) { d_needsResult = true; }
    }

    /** 
     * Set initialization routine association cache. 
     *
     * @param[in] inInit  Pass true if the INIT annotation is detected for the
     *                    routine; otherwise, pass false.
     */
    void setInInit(bool inInit) { d_isInInit = inInit; }

    /** Return whether the clause is associated with initialization routine. */
    bool isInInit() { return d_isInInit; }

    /** Return whether an assert clause. */
    bool isAssert() { return d_type == ContractComment_ASSERT; }

    /** Return whether a final comment. */
    bool isFinal() { return d_type == ContractComment_FINAL; }

    /** Return whether an init (or initialization) comment. */
    bool isInit() { return d_type == ContractComment_INIT; }

    /** Return whether an invariant clause. */
    bool isInvariant() { return d_type == ContractComment_INVARIANT; }

    /** 
     * Cache whether the 'is pure' annotation is included in the clause.
     *
     * @param[in] isPure  Pass true if the annotation is included in the clause;
     *                    otherwise, pass false.
     */
    void setIsPure(bool isPure) { d_isPure = isPure; }

    /** Return whether the clause contains an 'is pure' annotation. */
    bool isPure() { return d_isPure; }

    /** Return whether a precondition clause. */
    bool isPreconditions() { return d_type == ContractComment_PRECONDITION; }

    /** Return whether a postcondition clause. */
    bool isPostconditions() { return d_type == ContractComment_POSTCONDITION; }

    /** Return whether a dump statistics comment. */
    bool isStats() { return d_type == ContractComment_STATS; }

    /** 
     * Set return result assertion cache. 
     *
     * @param[in] needs  Pass true if the return variable is detected in one
     *                     or more of the assertion expressions; otherwise,
     *                     pass false.
     */
    void setResult(bool needs) { d_needsResult = needs; }

    /** Return whether a result variable needs to be generated. */
    bool needsResult() { return d_needsResult; }

    /** 
     * Return the comment for STATS; otherwise, return empty string. 
     * \warning {
     *  It is assumed the (first) expression corresponds to a comment.
     * }
     */
    std::string getComment() 
    {
      return (d_type == ContractComment_STATS) && (d_aeList.size() == 1) ? 
               d_aeList.front().expr() : "";
    }

    /** Return the list of assertion expressions. */
    std::list<AssertionExpression> getList() { return d_aeList; }

    /** 
     * Return the filename for INIT; otherwise, return empty string. 
     * \warning {
     * It is assumed the (first) expression corresponds to a filename.
     * }
     */
    std::string getFilename() 
    {
      return (d_type == ContractComment_INIT) && (d_aeList.size() == 1) ? 
               d_aeList.front().expr() : "";
    }

    /** Clear the list of assertion expressions. */
    void clear() { d_aeList.clear(); }

    /** Return the number of assertion expressions. */
    int size() { return d_aeList.size(); }

    /** Return the type of preprocessing directive. */
    PPIDirectiveType directive() { return d_dirType; }

    /** Return the number of executable assertion expressions in the clause. */
    int numExecutable() { return d_numExec; }

    /** 
     * Return a string representation of the contract comment.
     *
     * @param[in] sep  Field separator.
     * @return         Field-separated string representation.
     */
    std::string str(std::string sep)
    {
      std::ostringstream rep;
      switch (d_type)
      {
      case ContractComment_NONE:
        rep << "None";
        break;
      case ContractComment_INVARIANT:
        rep << "Invariant";
        break;
      case ContractComment_PRECONDITION:
        rep << "Precondition";
        break;
      case ContractComment_POSTCONDITION:
        rep << "Postcondition";
        break;
      case ContractComment_ASSERT:
        rep << "Assertion";
        break;
      case ContractComment_INIT:
        rep << "Init";
        break;
      case ContractComment_FINAL:
        rep << "Final";
        break;
      case ContractComment_STATS:
        rep << "Stats";
        break;
      default:
        rep << "UNKNOWN";
        break;
      }
      rep << sep;
      rep << d_aeList.size() << sep;
      rep << (d_needsResult ? "NeedsResult" : "NoResult") << sep;
      rep << (d_isInInit ? "isInInit" : "notInInit") << sep;
      rep << (d_isPure ? "isPure" : "notIsPure") << sep;
      rep << d_numExec;
      return rep.str();
    }


  private:
    /** Type of contract comment. */
    ContractCommentEnum             d_type;

    /** Assertion expressions associated with the contract comment. */
    std::list<AssertionExpression>  d_aeList;

    /** The type of preprocessing directive associated with the AST node. */
    PPIDirectiveType                d_dirType;

    /** Cache the number of executable expressions in the clause. */
    int                             d_numExec;

    /** Cache whether the function result appears in the clause. */
    bool                            d_needsResult;

    /** 
     * Cache whether the clause is associated with initialization routine. 
     *
     * This information is determines whether precondition checks are added
     * to the method.
     */
    bool                            d_isInInit;

    /** Cache whether the clause contains an 'is pure' annotation. */
    bool                            d_isPure;
};  /* class ContractComment */

/**
 * ContractClauseType consists of a vector of ContractComment entries.
 */
typedef std::vector<ContractComment*> ContractClauseType;


#endif /* include_Contract_Comment_hpp */
