/**
 * \internal
 * File:           ContractComment.hpp
 * Author:         T. Dahlgren
 * Created:        2012 November 9
 * Last Modified:  2013 January 31
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

using namespace std;


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
  /** INIT:  An initialization clause comment. */
  ContractComment_INIT,
  /** FINAL:  A finalization clause comment. */
  ContractComment_FINAL
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
  /** INIT:  No corresponding contract clause. */
  ContractClause_NONE,
  /** FINAL:  No corresponding contract clause. */
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
 * @todo  Consider (someday) refactoring to recognize the distinction 
 *   between the types of contract comments (ie, those with and those
 *   without assertions).
 */
class ContractComment
{
  public:
    /** Constructor */
    ContractComment(ContractCommentEnum t, PPIDirectiveType dt): 
      d_type(t), d_needsResult(false), d_isInit(false), d_dirType(dt), 
      d_numExec(0) {}

    /** Destructor */
    ~ContractComment() { d_aeList.clear(); }

    /** Return the contract comment type. */
    ContractCommentEnum type() { return d_type; }

    /** Return the contract comment clause type. */
    ContractClauseEnum clause() { return ContractCommentClause[d_type]; }

    /** 
     * Add the assertion expression to the contract clause. 
     *
     * @param[in] ae  Current assertion expression
     */
    void add(AssertionExpression ae) 
    { 
      d_aeList.push_front(ae); 
      if (ae.support() == AssertionSupport_EXECUTABLE) d_numExec += 1;
    }

    /** 
     * Set initialization routine association cache. 
     *
     * @param[in] init  Pass true if the INIT annotation is detected for the
     *                    routine; otherwise, pass false.
     */
    void setInit(bool init) { d_isInit = init; }

    /** Return whether the clause is associated with initialization routine. */
    bool isInit() { return d_isInit; }

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

    /** Return the list of assertion expressions. */
    list<AssertionExpression> getList() { return d_aeList; }

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
    string str(string sep)
    {
      ostringstream rep;
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
      case ContractComment_INIT:
        rep << "Init";
        break;
      case ContractComment_FINAL:
        rep << "Final";
        break;
      default:
        rep << "UNKNOWN";
        break;
      }
      rep << d_aeList.size() << sep;
      rep << (d_needsResult ? "NeedsResult" : "NoResult") << sep;
      rep << (d_isInit ? "isINIT" : "notINIT") << sep;
      rep << d_numExec;
      return rep.str();
    }


  private:
    /** Type of contract comment. */
    ContractCommentEnum        d_type;

    /** Assertion expressions associated with the contract comment. */
    list<AssertionExpression>  d_aeList;

    /** The type of preprocessing directive associated with the AST node. */
    PPIDirectiveType           d_dirType;

    /** Cache whether the function result appears in the clause. */
    bool                       d_needsResult;

    /** Cache whether the clause is associated with initialization routine. */
    bool                       d_isInit;

    /** Cache the number of executable expressions in the clause. */
    int                        d_numExec;
};  /* class ContractComment */

/**
 * ContractClauseType consists of a vector of ContractComment entries.
 */
typedef std::vector<ContractComment*> ContractClauseType;


#endif /* include_Contract_Comment_hpp */
