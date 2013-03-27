/**
 * \internal
 * File:           ContractsProcessor.cpp
 * Author:         T. Dahlgren
 * Created:        2012 November 1
 * Last Modified:  2013 March 26
 * \endinternal
 *
 * @file
 * @brief
 * Basic contract clause processing utilities.
 *
 * @htmlinclude copyright.html
 */

#include <iostream>
#include <sstream>
#include <list>
#include <string>
#include "rose.h"
#include "Cxx_Grammar.h"
#include "RoseHelpers.hpp"
#include "contractOptions.h"
#include "contractClauseTypes.hpp"
#include "ContractsProcessor.hpp"

using namespace std;


const string S_PREFACE = "PContract: ";

#ifdef PCE_ADD_COMMENTS
const string S_BEGIN = ": BEGIN"; 
const string S_END = ": END"; 

const string S_INCLUDES = "Includes";

const string S_DUMP = "Data Dump";
const string S_FINAL = "Finalization";
const string S_INIT = "Initialization";
#endif /* PCE_ADD_COMMENTS */

const string S_ERROR = "Error: Contract Developer(s)";
const string S_ASSERTIONS = "Assertions";
const string S_INVARIANTS = "Invariants";
const string S_PRECONDITIONS = "Preconditions";
const string S_POSTCONDITIONS = "Postconditions";


const string S_ERROR_NOT_SCOPED = 
  "ERROR: The contract annotation MUST be associated with a scoped statement.";

const string S_ERROR_ASSERTIONS = 
  "ERROR: Assertions CANNOT be associated with function definitions.";
const string S_ERROR_POSTCONDITIONS = 
  "ERROR: Postconditions MUST be associated with function definitions.";
const string S_ERROR_PRECONDITIONS = 
  "ERROR: Preconditions MUST be associated with function definitions.";

/*
 * Would prefer to use the following to add a comment or at least a C-style
 * message; however, that appears to cause problems ranging from SEGV to
 * non-comment text.  
 *
const string S_NOTE_INVARIANTS = 
  "NOTE: (Class) Invariants are NOT supported for non-instance routines.";
 *
 * So modified the note to include C-style comments AND formatting (under 
 * the assumption it is necessary to simply add the comment as text for the 
 * unparser).
 */
const string S_NOTE_INVARIANTS = 
  "\n/* NOTE: Invariants are NOT supported for non-instance routines. */\n";

const string S_WARN_INVARIANTS = 
  "WARNING: Ignoring additional invariant clause.";

const string S_WARN_MULTI_NOFUNC = 
  "WARNING: Multiple annotations detected for non-function node.";

const string S_WARN_NO_EXECS = 
  "WARNING: Expecting first clause but no executable contract checks.";

const string S_SEP = ";";


/**
 * Build a contracts stats dump statement.
 *
 * @param[in] currSttmt  The current AST statement node.
 * @param[in] dt         The pre-processing directive type.
 * @return               The function call statement or NULL.
 */
SgExprStatement*
buildDump(
  /* in */ SgStatement*      currSttmt,
  /* in */ PPIDirectiveType  dt)
{
  SgExprStatement* sttmt = NULL;

  SgExprListExp* parms = new SgExprListExp(FILE_INFO);
  if ( (currSttmt != NULL) && (parms != NULL) )
  {
    parms->append_expression(SageBuilder::buildVarRefExp("pce_enforcer"));
    parms->append_expression(new SgStringVal(FILE_INFO, "End processing"));
    sttmt = SageBuilder::buildFunctionCallStmt("PCE_DUMP_STATS", 
      SageBuilder::buildVoidType(), parms, currSttmt->get_scope());
    if (sttmt != NULL)
    {
#ifdef PCE_ADD_COMMENTS
      SageInterface::attachComment(sttmt, S_PREFACE + S_DUMP + S_BEGIN,
        PreprocessingInfo::before, dt);
      SageInterface::attachComment(sttmt, S_PREFACE + S_DUMP + S_END,
        PreprocessingInfo::after, dt);
#endif /* PCE_ADD_COMMENTS */
    }
  }

  return sttmt;
} /* buildDump */


/**
 * Build a contracts finalization statement.
 *
 * @param[in] currSttmt  The current AST statement node.
 * @param[in] dt         The pre-processing directive type.
 * @return               The function call statement or NULL.
 */
SgExprStatement*
buildFinal(
  /* in */ SgStatement*      currSttmt,
  /* in */ PPIDirectiveType  dt)
{
  SgExprStatement* sttmt = NULL;

  SgExprListExp* parms = new SgExprListExp(FILE_INFO);
  if ( (currSttmt != NULL) && (parms != NULL) )
  {
    sttmt = SageBuilder::buildFunctionCallStmt("PCE_FINALIZE", 
      SageBuilder::buildVoidType(), parms, currSttmt->get_scope());
    if (sttmt != NULL)
    {
#ifdef PCE_ADD_COMMENTS
      SageInterface::attachComment(sttmt, S_PREFACE + S_FINAL + S_BEGIN,
        PreprocessingInfo::before, dt);
      SageInterface::attachComment(sttmt, S_PREFACE + S_FINAL + S_END,
        PreprocessingInfo::after, dt);
#endif /* PCE_ADD_COMMENTS */
    }
  }

  return sttmt;
} /* buildFinal */


/**
 * Build a contracts initialization statement.
 *
 * @param[in] currSttmt  The current AST statement node.
 * @param[in] dt         The pre-processing directive type.
 * @return               The function call statement or NULL.
 */
SgExprStatement*
buildInit(
  /* in */ SgStatement*      currSttmt,
  /* in */ PPIDirectiveType  dt)
{
  SgExprStatement* sttmt = NULL;

  SgExprListExp* parms = new SgExprListExp(FILE_INFO);
  if ( (currSttmt != NULL) && (parms != NULL) )
  {
    parms->append_expression(SageBuilder::buildVarRefExp("NULL"));
    sttmt = SageBuilder::buildFunctionCallStmt("PCE_INITIALIZE", 
      SageBuilder::buildVoidType(), parms, currSttmt->get_scope());
    if (sttmt != NULL)
    {
#ifdef PCE_ADD_COMMENTS
      SageInterface::attachComment(sttmt, S_PREFACE + S_INIT + S_BEGIN,
        PreprocessingInfo::before, dt);
      SageInterface::attachComment(sttmt, S_PREFACE + S_INIT + S_END,
        PreprocessingInfo::after, dt);
#endif /* PCE_ADD_COMMENTS */
    }
  }

  return sttmt;
} /* buildInit */


/**
 * Extract and add assertion expressions to the contract clause.
 *
 * @param[in] clause          The contract clause text extracted from the 
 *                              structured comment.
 * @param     cc              [inout] The resulting contract clause/comment.
 * @param[in] firstExecClause Expected to be the first executable clause.
 */
void
ContractsProcessor::addExpressions(
  /* in */    string           clause, 
  /* inout */ ContractComment* cc, 
  /* in */    bool             firstExecClause)
{
  if (!clause.empty() && cc != NULL)
  {
    size_t startAE = 0, endAE;
    bool isFirst = firstExecClause;
    while ( (endAE=clause.find(";", startAE)) != string::npos )
    {
      string statement = clause.substr(startAE, endAE-startAE);
      if (!statement.empty())
      {
        string label, expr;
        size_t startE = 0, endL;

#ifdef DEBUG
        cout << "DEBUG: Extracted: " ;
#endif /* DEBUG */

        if ( (endL=statement.find(":")) != string::npos )
        {
          if (statement[endL+1] != ':') {
            label = compress(statement.substr(0, endL));
            startE = endL+1;
#ifdef DEBUG
            cout << label + ": ";
#endif /* DEBUG */
          }
        }

        expr = compress(statement.substr(startE));

#ifdef DEBUG
        cout << expr << "\n";
#endif /* DEBUG */

        if (expr.find("pce_result") != string::npos )
        {
          cc->setResult(true);
#ifdef DEBUG
        cout << "DEBUG: ..contains pce_result\n";
#endif /* DEBUG */
        }

        if (expr == "is pure") 
        {
#ifdef DEBUG
            cout << "DEBUG: ..is advisory expression.\n";
#endif /* DEBUG */
        }
        else if (expr == "is initialization") 
        {
          cc->setInInit(true);
#ifdef DEBUG
            cout << "DEBUG: ..is initialization routine.\n";
#endif /* DEBUG */
        } 
        else if (isExecutable(expr))
        {
          AssertionExpression ae (label, expr, AssertionSupport_EXECUTABLE,
            isFirst);
          cc->add(ae);
          isFirst = false;
#ifdef DEBUG
            cout << "DEBUG: ..is executable expression.\n";
#endif /* DEBUG */
        } 
        else
        {
          AssertionExpression ae (label, expr, AssertionSupport_UNSUPPORTED,
            false);
          cc->add(ae);
#ifdef DEBUG
            cout << "DEBUG: ..includes an unsupported keyword expression.\n";
#endif /* DEBUG */
        } 

      }

      startAE = endAE+1;
    }
  }

  return;
} /* addExpressions */


/**
 * Build and add contract enforcement finalization call.
 *
 * @param def  Function definition.
 * @param body Function body.
 * @param cc   (FINAL) Contract comment.
 * @return     Returns number of finalization calls added.    
 */
int
ContractsProcessor::addFinalize(SgFunctionDefinition* def, SgBasicBlock* body, 
  ContractComment* cc)
{
  int num = 0;

  if ( (def != NULL) && (body != NULL) && (cc != NULL) )
  {
    SgExprStatement* sttmt;

    PPIDirectiveType dt = cc->directive();

#ifdef PCE_ADD_STATS_DUMP
    /*
     * @todo TBD/FIX: Add statistics dump?
     */
    sttmt = buildFinal(body, dt);
    if (sttmt != NULL)
    {
      num += SageInterface::instrumentEndOfFunction(def->get_declaration(), 
                                                    sttmt);
    }
#endif /* PCE_ADD_STATS_DUMP */

    sttmt = buildFinal(body, dt);
    if (sttmt != NULL)
    {
      num += SageInterface::instrumentEndOfFunction(def->get_declaration(),
                                                    sttmt);
    }
  }
              
  return num;
}  /* addFinalize */


/**
 * Add requisite include file(s).
 *
 * @param globalScope  The Sage project representing the initial AST of the
 *                       file(s).
 * @return             The processing status: 0 for success, non-0 for failure.
 */
int
ContractsProcessor::addIncludes(
  /* inout */ SgGlobal* globalScope)
{
  int status = 0;

  if ( globalScope != NULL )
  {
#ifdef PCE_ADD_COMMENTS
    SageInterface::attachComment(globalScope, S_PREFACE + S_INCLUDES,
      PreprocessingInfo::before, PreprocessingInfo::C_StyleComment);
#endif /* PCE_ADD_COMMENTS */

    SageInterface::addTextForUnparser(globalScope, S_INCLUDES,
      AstUnparseAttribute::e_after);

    /*
     * Assuming sufficient to determine if exit (for stdlib.h) is
     * defined to know if one of the include files is present.  A
     * bit of an inefficient hack, but the check is only done once
     * per file.
     */
    string decls = globalScope->unparseToString();
    if (decls.find("void exit") == string::npos)
    {
      SageInterface::addTextForUnparser(globalScope, "\n#include <stdlib.h>",
        AstUnparseAttribute::e_before);
    }
  }
  else
  {
    cerr<<"\nSkipping addition of includes due to lack of global scope.\n";
    status = 1;
  }

  return status;
}  /* addIncludes */


/**
 * Add requisite include file(s).
 *
 * @param project         The Sage project representing the initial AST of the
 *                          file(s).
 * @param skipTransforms  True if the transformations are to be skipped;
 *                          otherwise, false.
 * @return                The processing status: 0 for success, non-0 for 
 *                          failure.
 */
int
ContractsProcessor::addIncludes(SgProject* project, bool skipTransforms)
{
  int status = 0;

  if (project != NULL)
  {
    Rose_STL_Container<SgNode*> globalScopeList = 
      NodeQuery::querySubTree(project, V_SgGlobal);
    for (Rose_STL_Container<SgNode*>::iterator i = globalScopeList.begin();
         i != globalScopeList.end(); i++)
    {
      Sg_File_Info* info;
      SgGlobal* globalScope;
      if ( (  (globalScope = isSgGlobal(*i)) != NULL)
           && ((info=globalScope->get_file_info()) != NULL)
           && isInputFile(project, info->get_raw_filename()) )
      {
        if (skipTransforms)
        {
          cout<<"\naddIncludes: "<<info->get_raw_filename()<<"\n";
        }
        else
        {
          status = addIncludes(globalScope);
        }
      }
    }
  }
  else
  {
    cerr<<"\nSkipping addition of includes due to lack of project.\n";
    status = 1;
  }

  return status;
}  /* addIncludes */


/**
 * Build and add the contract enforcement initialization call.
 *
 * @param body   Pointer to the function body.
 * @param cc     (INIT) Contract comment.
 * @return       Number of initialization calls added.
 */
int
ContractsProcessor::addInitialize(SgBasicBlock* body, ContractComment* cc)
{
  int num = 0;

  if ( (body != NULL) && (cc != NULL) )
  {
    SgExprStatement* sttmt = buildInit(body, cc->directive());
    if (sttmt != NULL) {
      body->prepend_statement(sttmt);
      num += 1;
    }
  }
              
  return num;
}  /* addInitialize */


/**
 * Add checks for all contract clause assertion expressions to the end
 * of the routine body.  
 *
 * @param     def  [inout] Function definition.
 * @param     body [inout] Pointer to the function body, which is assumed
 *                   to belong to the function definition, def.
 * @param[in] cc   The contract comment whose expressions are to be added.
 * @return         The number of statements added to the body.
 */
int
ContractsProcessor::addPostChecks(
  /* inout */ SgFunctionDefinition* def, 
  /* inout */ SgBasicBlock*         body, 
  /* in */    ContractComment*      cc)
{
  int num = 0;

  if ( (def != NULL) && (body != NULL) && (cc != NULL) && (cc->size() > 0) )
  {
    ContractClauseEnum ccType = cc->clause();
    if (  (ccType == ContractClause_POSTCONDITION)
       || (ccType == ContractClause_INVARIANT) )
    {
#ifdef DEBUG
      cout << "DEBUG: addPostChecks: Adding "<<cc->size();
      cout << " expressions and/or comments...\n";
#endif /* DEBUG */

      list<AssertionExpression> aeList = cc->getList();
      for(list<AssertionExpression>::iterator iter = aeList.begin();
          iter != aeList.end(); iter++)
      {
        AssertionExpression ae = (*iter);
        switch (ae.support())
        {
          case AssertionSupport_EXECUTABLE:
            {
              SgExprStatement* sttmt = buildCheck(body, ccType, ae, 
                                                  cc->directive());
              if (sttmt != NULL)
              {
                num += SageInterface::instrumentEndOfFunction(
                         def->get_declaration(), sttmt);
              }
            }
            break;
          case AssertionSupport_UNSUPPORTED:
            {
#ifdef DEBUG
              cout << "DEBUG: ..unsupported expression(s)\n";
#endif /* DEBUG */
              /*
               * Attach unsupported comment (NOTE: ROSE/unparser seems to 
               * ignore!)
               */
              SageInterface::attachComment(body, 
                S_PREFACE + string(S_CONTRACT_CLAUSE[ccType])
                  + ": " + L_UNSUPPORTED_EXPRESSION + ae.expr(),
                PreprocessingInfo::after, cc->directive());
            }
            break;
          default:
            // Nothing to do here
#ifdef DEBUG
            cout << "DEBUG: ..unrecognized support level\n";
#endif /* DEBUG */
            break;
        }
      }

      d_first = false;
    } 
    else
    { 
      cerr<<"\nERROR: Refusing to add non-postcondition and non-invariant ";
      cerr<<"checks to routine end.\n";
    } /* end if have something to work with */

#ifdef DEBUG
    cout << "DEBUG: addPostChecks: number statements appended = "<<num<<"\n";
#endif /* DEBUG */
  } /* end if have something to work with */

  return num;
}  /* addPostChecks */


/**
 * Add checks for all contract clause assertion expressions to the start
 * of the routine body.  
 *
 * @param body    [inout] Pointer to the function body, which is assumed to 
 *                  belong to an SgFunctionDefinition node.
 * @param[in] cc  The contract clause/comment whose (executable) expressions
 *                  are to be added.
 * @return        The number of statements added to the body.
 */
int
ContractsProcessor::addPreChecks(
  /* inout */ SgBasicBlock*    body,
  /* in */    ContractComment* cc)
{
  int num = 0;

  if ( (body != NULL) && (cc != NULL) && (cc->size() > 0) )
  {
    ContractClauseEnum ccType = cc->clause();
    if (  (ccType == ContractClause_PRECONDITION)
       || (ccType == ContractClause_INVARIANT) )
    {
#ifdef DEBUG
      cout << "DEBUG: addPreChecks: Adding "<<cc->size();
      cout << " expressions and/or comments...\n";
#endif /* DEBUG */

      list<AssertionExpression> aeList = cc->getList();
      for(list<AssertionExpression>::iterator iter = aeList.begin();
          iter != aeList.end(); iter++)
      {
        AssertionExpression ae = (*iter);
        switch (ae.support())
        {
          case AssertionSupport_EXECUTABLE:
            {
              SgExprStatement* sttmt = buildCheck(body, ccType, ae, 
                cc->directive());
              if (sttmt != NULL)
              {
                body->prepend_statement(sttmt);
                num++;
              }
            }
            break;
          case AssertionSupport_UNSUPPORTED:
            {
              SageInterface::attachComment(body,
                S_PREFACE + string(S_CONTRACT_CLAUSE[ccType])
                  + ": " + L_UNSUPPORTED_EXPRESSION + ae.expr(),
                PreprocessingInfo::before, cc->directive());
            }
            break;
          default:
            // Nothing to do here
            break;
        }
      }
    } 
    else
    { 
      cerr<<"\nERROR: Refusing to add non-precondition and non-invariant ";
      cerr<<"checks to routine start.\n";
    } /* end if have something to work with */

#ifdef DEBUG
    cout << "DEBUG: addPreChecks: Number of statements prepended = "<<num<<"\n";
#endif /* DEBUG */
  } /* end if have something to work with */

  return num;
}  /* addPreChecks */


/**
 * Build the contract clause check statement.
 *
 * @param     currSttmt  [inout] Pointer to the current statement.
 * @param[in] clauseType The type of contract clause associated with the 
 *                         expression.
 * @param[in] ae         The assertion expression.
 * @param[in] dt         The (comment) directive type.
 * @return               Contract clause statement node.
 */
SgExprStatement*
ContractsProcessor::buildCheck(
  /* inout */ SgStatement*        currSttmt, 
  /* in */    ContractClauseEnum  clauseType, 
  /* in */    AssertionExpression ae, 
  /* in */    PPIDirectiveType    dt)
{
  SgExprStatement* sttmt = NULL;

  if ( (currSttmt != NULL) && (ae.support() == AssertionSupport_EXECUTABLE) )
  {
    string cmt, clauseTypeStr, clauseTime;
    SgExprListExp* parms = new SgExprListExp(FILE_INFO);

    if (parms != NULL)
    {
      parms->append_expression(SageBuilder::buildVarRefExp("pce_enforcer"));
  
      switch (clauseType)
      {
        case ContractClause_PRECONDITION:
          {
            clauseTypeStr = "ContractClause_PRECONDITION";
            clauseTime = "pce_def_times.pre";
            cmt = S_PRECONDITIONS;
          }
          break;
        case ContractClause_POSTCONDITION:
          {
            clauseTypeStr = "ContractClause_POSTCONDITION";
            clauseTime = "pce_def_times.post";
            cmt = S_POSTCONDITIONS;
          }
          break;
        case ContractClause_INVARIANT:
          {
            clauseTypeStr = "ContractClause_INVARIANT";
            clauseTime = "pce_def_times.inv";
            cmt = S_INVARIANTS;
          }
          break;
        case ContractClause_ASSERT:
          {
            clauseTypeStr = "ContractClause_ASSERT";
            clauseTime = "pce_def_times.asrt";
            cmt = S_ASSERTIONS;
          }
          break;
        default:
          {
            /*  WARNING: This should NEVER happen. */
            clauseTypeStr = "ContractClause_NONE";
            clauseTime = "0";
            cmt = S_ERROR;
          }
          break;
      }

      parms->append_expression(SageBuilder::buildVarRefExp(clauseTypeStr));
      parms->append_expression(SageBuilder::buildVarRefExp(clauseTime));
      parms->append_expression(SageBuilder::buildVarRefExp(
        "pce_def_times.routine"));
      parms->append_expression(SageBuilder::buildVarRefExp(
        (ae.isFirst() && d_first) ? "CONTRACTS_TRUE" : "CONTRACTS_FALSE"));
      parms->append_expression(new SgStringVal(FILE_INFO, ae.label()));
      parms->append_expression(SageBuilder::buildVarRefExp("("+ae.expr()+")"));
  
      sttmt = SageBuilder::buildFunctionCallStmt("PCE_CHECK_EXPR_TERM", 
        SageBuilder::buildVoidType(), parms, currSttmt->get_scope());

#ifdef PCE_ADD_COMMENTS
      if (sttmt != NULL)
      {
        SageInterface::attachComment(sttmt, S_PREFACE + cmt + S_BEGIN,
          PreprocessingInfo::before, dt);
        SageInterface::attachComment(sttmt, S_PREFACE + cmt + S_END,
          PreprocessingInfo::after, dt);
      }
#endif /* PCE_ADD_COMMENTS */
#ifdef DEBUG
      else
      {
        cout << "DEBUG: buildCheck: New statement is NULL.\n";
      }
#endif /* DEBUG */
    }
#ifdef DEBUG
    else
    {
      cout << "DEBUG: buildCheck: New parameter list is NULL.\n";
    }
#endif /* DEBUG */
  }
#ifdef DEBUG
  else
  {
    if (currSttmt == NULL) 
    {
      cout << "DEBUG: buildCheck: Given NULL body (wrong!)\n";
    }

    if (ae.support() != AssertionSupport_EXECUTABLE)
    {
      cout << "DEBUG: buildCheck: Passed non-executable expression (wrong!)\n";
    }
  }
#endif /* DEBUG */
              
  return sttmt;
}  /* buildCheck */


/**
 * Extract the contract, if any, associated with the node.
 *
 * @param lNode      [in] Current located AST node.
 * @param firstExec  [in] True if any associated clause is expected to be
 *                        the first with executable contracts; False 
 *                        otherwise.
 * @param clauses    [inout] The contract, which may consist of one or more 
 *                           clauses.
 */
void
ContractsProcessor::extractContract(
  /* in */    SgLocatedNode*     lNode,
  /* in */    bool               firstExec,
  /* inout */ ContractClauseType &clauses)
{
  int num = 0;

  if (lNode != NULL)
  {
#ifdef DEBUG
//    printLineComment(lNode, "DEBUG: ..extracting node comments.", false);
#endif /* DEBUG */

    AttachedPreprocessingInfoType* cmts = 
      lNode->getAttachedPreprocessingInfo();

    if (cmts != NULL)
    {
#ifdef DEBUG
//      printLineComment(lNode, "DEBUG: ....processing attached comments", false);
#endif /* DEBUG */

      AttachedPreprocessingInfoType::iterator iter;
      for (iter = cmts->begin(); iter != cmts->end(); iter++)
      {
        ContractComment* cc = extractContractComment(lNode, iter, firstExec);
        if (cc != NULL)
        {
          if (firstExec && (cc->numExecutable() > 0))
          {
            firstExec = false;
          }

          clauses.push_back(cc);
        } /* end if have contract comment to process */
      } /* end for each comment */

    } /* end if have comments */
#ifdef DEBUG
/*
    else
    {
      cout<<"DEBUG: ....no attached comments\n";
    }
*/
#endif /* DEBUG */
  } /* end if have a node */

#ifdef DEBUG
  if (num > 1)
  {
    cout<<"DEBUG: ....multiple annotations detected.\n";
  }
#endif /* DEBUG */

  return;
}  /* extractContract */


/**
 * Extract the contract clause comment, if any, from the pre-processing 
 * directive.
 *
 * @param[in] aNode           Current AST node.
 * @param[in] info            The preprocessing directive.
 * @param[in] firstExecClause Expected to be the first executable clause.
 * @return                    The ContractComment type.
 */
ContractComment*
ContractsProcessor::extractContractComment(
  /* in */ SgNode*                                 aNode, 
  /* in */ AttachedPreprocessingInfoType::iterator info, 
  /* in */ bool                                    firstExecClause)
{
  ContractComment* cc = NULL;

  if ((*info) != NULL)
  {
    PreprocessingInfo::DirectiveType dt = (*info)->getTypeOfDirective();
    switch (dt)
    {
      case PreprocessingInfo::C_StyleComment:
        {
          string str = (*info)->getString();
          cc = processCommentEntry(aNode, str.substr(2, str.size()-4), dt,
                 firstExecClause);
        }
        break;
      case PreprocessingInfo::CplusplusStyleComment:
        {
          string str = (*info)->getString();
          cc = processCommentEntry(aNode, str.substr(2), dt, firstExecClause);
        }
        break;
      default:
        /* Nothing to do here */
        break;
    }
  }
#ifdef DEBUG
  else
  {
    cout << "DEBUG: extractContractComment: Information is NULL\n";
  }
#endif /* DEBUG */

  return cc;
}  /* extractContractComment */


/**
 * Determine if what is assumed to be the method name is in the 
 * invariants clause.
 *
 * @param[in] nm  Method name.
 * @return        True if nm is in at least one invariant expression; false 
 *                  otherwise.
 */
bool
ContractsProcessor::inInvariants(
  /* in */ string nm)
{
  bool isIn = false;
  
  if ( !nm.empty() && (d_invariants != NULL) )
  {
    list<AssertionExpression> aeList = d_invariants->getList();
    for(list<AssertionExpression>::iterator iter = aeList.begin();
        iter != aeList.end() && !isIn; iter++)
    {
      AssertionExpression ae = (*iter);
      if (ae.support() == AssertionSupport_EXECUTABLE)
      {
        string expr = ae.expr();
        if ( !expr.empty() && expr.find(nm) != string::npos )
        {
          isIn = true;
        }
      }
    }
  }

  return isIn;
}  /* inInvariants */


/**
 * Add (test) contract assertion checks to each routine.
 *
 * @param project             [inout] The Sage project representing the initial
 *                              AST of the file(s).
 * @param[in] skipTransforms  True if the transformations are to be skipped;
 *                              otherwise, false.
 * @return                    The processing status: 0 for success, non-0 for 
 *                              failure.
 */
int
ContractsProcessor::instrumentRoutines(
  /* inout */ SgProject* project, 
  /* in */    bool       skipTransforms)
{
  int status = 0;

  if (project != NULL)
  {
    /* Find all function definitions. */
    vector<SgNode*> fdList = 
      NodeQuery::querySubTree(project, V_SgFunctionDefinition);

    if (!fdList.empty())
    {
      int num = 0;

      vector<SgNode*>::iterator iter;
      for (iter=fdList.begin(); iter!=fdList.end(); iter++)
      {
        Sg_File_Info* info;
        SgFunctionDefinition* def = isSgFunctionDefinition(*iter);

        /*
         * Get all of the attendant information and ensure the node
         * is ACTUALLY to be associated with generated output.
         */
        if (  (def != NULL) 
           && ((info=def->get_file_info()) != NULL)
           && isInputFile(project, info->get_raw_filename()) )
        {
#ifdef DEBUG
            printLineComment(def, "DEBUG: Have a function definition node.",
                             false);
#endif /* DEBUG */

          if (skipTransforms)
          {
            cout<<"\n"<<++num<<": "<<info->get_raw_filename()<<":\n   ";
            cout<<getBasicSignature(def)<<endl;
          }
          else
          {
            num += processFunctionDef(def);
          }
        }
      }

      cout<<"Added "<<num<<" contract-related statements.\n";
    }

    /* Run consistency checks (?) */
    //AstTests::runAllTests(project);

    /* Translate the file(s) */
    project->unparse();

#ifdef DEBUG
    cout<<"Processed "<<fdList.size()<<" function definitions.\n\n";
#endif /* DEBUG */
  }
  else
  {
    cerr<<"\nSkipping routines instrumentation to lack of project.\n";
    status = 1;
  }

  return status;
}  /* instrumentRoutines */


/**
 * Determines whether the expression APPEARS to be executable in C/C++.  There 
 * are no syntactic or semantic checks other than to eliminate expressions
 * known not to translate.
 *
 * @param[in] expr  The string representing the assertion expression.
 * @return          True if the expression appears to be executable; False 
 *                    otherwise.
 */
bool
ContractsProcessor::isExecutable(
  /* in */string expr)
{
  bool isOkay = true;

  if (!expr.empty())
  {
    for (int i=MIN_NEE_INDEX; i<MAX_NEE_INDEX; i++)
    {
      if (expr.find(ReservedWordPairs[i][0]) != string::npos)
      {
#ifdef DEBUG
        cout << "DEBUG: Detected \'" << ReservedWordPairs[i][0];
        cout << "\' in \'"<< expr << "\' making expression non-executable.\n";
#endif /* DEBUG */
        isOkay = false;
        break;
      }
    }
  }

  return isOkay;
}  /* isExecutable */


/**
 * Process the comment to assess and handle any contract annotation.
 *
 * @param     aNode           [inout] Current AST node.
 * @param[in] cmt             Comment contents.
 * @param[in] dirType         (Comment) directive type.
 * @param[in] firstExecClause Expected to be the first executable clause.
 * @return                The corresponding ContractComment type.
 */
ContractComment*
ContractsProcessor::processCommentEntry(
  /* inout */ SgNode*                          aNode, 
  /* in */    string                           cmt, 
  /* in */    PreprocessingInfo::DirectiveType dirType, 
  /* in */    bool                             firstExecClause)
{
  ContractComment* cc = NULL;

  if ( (aNode != NULL) && !cmt.empty() )
  {
    size_t pos;
    if ((pos=cmt.find("CONTRACT"))!=string::npos)
    {
      if ((pos=cmt.find("REQUIRE"))!=string::npos)
      {
        cc = new ContractComment(ContractComment_PRECONDITION, dirType);
        addExpressions(cmt.substr(pos+7), cc, firstExecClause);
#ifdef DEBUG
        cout<<"DEBUG: Created REQUIRE ContractComment: "<<cc->str(S_SEP)<<endl;
#endif /* DEBUG */
      }
      else if ((pos=cmt.find("ENSURE"))!=string::npos)
      {
        cc = new ContractComment(ContractComment_POSTCONDITION, dirType);
        addExpressions(cmt.substr(pos+6), cc, firstExecClause);
#ifdef DEBUG
        cout<<"DEBUG: Created ENSURE ContractComment: "<<cc->str(S_SEP)<<endl;
#endif /* DEBUG */
      }
      else if ((pos=cmt.find("INVARIANT"))!=string::npos)
      {
        cc = new ContractComment(ContractComment_INVARIANT, dirType);
        addExpressions(cmt.substr(pos+9), cc, firstExecClause);
#ifdef DEBUG
        cout<<"DEBUG: Created INVARIANT ContractComment: ";
        cout<<cc->str(S_SEP)<<endl;
#endif /* DEBUG */
      }
      else if ((pos=cmt.find("ASSERT"))!=string::npos)
      {
        cc = new ContractComment(ContractComment_ASSERT, dirType);
        addExpressions(cmt.substr(pos+6), cc, firstExecClause);
//#ifdef DEBUG
        cout<<"DEBUG: Created ASSERT ContractComment: ";
        cout<<cc->str(S_SEP)<<endl;
//#endif /* DEBUG */
      }
      else if ((pos=cmt.find("INIT"))!=string::npos)
      {
        cc = new ContractComment(ContractComment_INIT, dirType);
#ifdef DEBUG
        cout<<"DEBUG: Created INIT ContractComment: "<<cc->str(S_SEP)<<endl;
#endif /* DEBUG */
      }
      else if ((pos=cmt.find("FINAL"))!=string::npos)
      {
        cc = new ContractComment(ContractComment_FINAL, dirType);
#ifdef DEBUG
        cout<<"DEBUG: Created FINAL ContractComment: "<<cc->str(S_SEP)<<endl;
#endif /* DEBUG */
      }
      else
      {
        string msg = "WARNING: Unidentified contract annotation: ";
        printLineComment(aNode, msg + cmt.substr(pos+8), false);
      }
    }
  }
  return cc;
} /* processCommentEntry */


/**
 * Process comments associated with the function definition.
 *
 * @param def  [inout] The function definition node.
 * @return     The number of statements added.
 */
int
ContractsProcessor::processFunctionComments(
  /* inout */ SgFunctionDefinition* def)
{
  int num = 0;

  if (def != NULL)
  {
    SgFunctionDeclaration* decl = def->get_declaration();
    if (decl != NULL)
    {
#ifdef DEBUG
      cout<<"DEBUG: ..obtained function declaration\n";
#endif /* DEBUG */

      bool isConstructor = false;
      bool isDestructor = false;
      bool isInitRoutine = false;
      bool isMemberFunc = false;
      SgMemberFunctionDeclaration* mfDecl = 
        isSgMemberFunctionDeclaration(decl);
      if (mfDecl != NULL)
      {
        SgSpecialFunctionModifier sfMod = 
          mfDecl->get_specialFunctionModifier();
        isConstructor = sfMod.isConstructor();
        isDestructor = sfMod.isDestructor();
        isMemberFunc = true;
      }

      SgName nm = decl->get_name();

      ContractClauseType clauses;
      extractContract(decl, true, clauses);
#ifdef DEBUG
      cout<<"DEBUG: ....extracted "<<clauses.size()<<" contract clause(s).\n";
#endif /* DEBUG */

      if (clauses.size() > 0)
      {
        ContractComment* final = NULL;
        ContractComment* init = NULL;
        ContractComment* pre = NULL;
        ContractComment* post = NULL;
        int numChecks[] = { 0, 0, 0 };
        int numPrep[] = { 0, 0 };
        bool firstExecClause = true;

        ContractClauseType::iterator iter;
        for (iter = clauses.begin(); iter != clauses.end(); iter++)
        {
          ContractComment* cc = (*iter);
          if (cc != NULL)
          {
#ifdef PCE_ENABLE_WARNING
            if (firstExecClause && (cc->numExecutable() <= 0))
            {
              printLineComment(def, S_WARN_NO_EXECS, false);
            }
#endif /* PCE_ENABLE_WARNING */

            switch (cc->type())
            {
            case ContractComment_PRECONDITION:
              {
                pre = cc;
                if (cc->isInInit()) isInitRoutine = true;
                numChecks[0] += pre->size();
              }
              break;
            case ContractComment_POSTCONDITION:
              {
                post = cc;
                numChecks[1] += post->size();
              }
              break;
            case ContractComment_INVARIANT:
              {
                setInvariants(isSgLocatedNode(def), cc);
#if 0
/* This is the current working version for a function definition. */
                /* 
                 * The following adds a C comment to .cc files BUT simple
                 * text to .cpp files.
                 *
                SageInterface::addMessageStatement(def, S_NOTE_INVARIANTS);
                 */
                SageInterface::addTextForUnparser(def, S_NOTE_INVARIANTS,
                  AstUnparseAttribute::e_before);

                if (d_invariants == NULL)
                {
                  d_invariants = cc;
                }
                else
                {
                  printLineComment(decl, S_WARN_INVARIANTS, false);
                  cc = NULL;
                }
#endif
              }
              break;
            case ContractComment_INIT:
              {
                init = cc;
              }
              break;
            case ContractComment_FINAL:
              {
                final = cc;
              }
              break;
            case ContractComment_ASSERT:
              {
//#ifdef DEBUG
                cout<<"DEBUG: ......erroneous ASSERT clause\n";
//#endif /* DEBUG */

                printLineComment(decl, S_ERROR_ASSERTIONS, false);
              }
              break;
            case ContractComment_NONE:
            default:
              {
                /* Nothing to do here. */
              }
              break;
            } /* end switch */
          } /* end if have contract comment to process */
        } /* end for each comment */

        if (  (init != NULL) || (final != NULL) || (pre != NULL) 
           || (post != NULL) || (d_invariants != NULL) )
        {
          SgBasicBlock* body = def->get_body();
  
          if (body != NULL)
          {
            if (d_invariants != NULL)
            {
              numChecks[2] = d_invariants->size();
            }
            /*
             * First add initial routine instrumentation.
             * ..Order IS important since each is prepended to the body.
             */
            bool skipInvariants = (nm=="main") || (init!=NULL) || (final!=NULL)
              || (d_invariants==NULL) || (numChecks[2]<=0) || isConstructor 
              || (!isMemberFunc) || inInvariants(nm);

            bool havePreChecks = (pre != NULL) && (numChecks[0] > 0);

            bool fixFirst = havePreChecks 
              && (pre->numExecutable() > 0) && d_first;

            if (! (skipInvariants || isInitRoutine) )
            {
//#ifdef DEBUG
              cout << "DEBUG: Invariants: pre->numExecutable()=";
              cout << pre->numExecutable() << ", first=" << d_first << endl;
//#endif /* DEBUG */

              if (fixFirst) d_first = false;
              num += addPreChecks(body, d_invariants);
              if (fixFirst) d_first = true;
            }

            if (havePreChecks)
            { 
              num += addPreChecks(body, pre);
              if (fixFirst) d_first = false;
            }

            if (init != NULL)
            {
              numPrep[0] += addInitialize(body, init);
              num += 1;
            }

            /*
             * Now add post-routine checks.
             */
            if (post != NULL) 
            {
              if (numChecks[1] > 0)
              {
                num += addPostChecks(def, body, post);
              }
            }

            if (! (skipInvariants || isDestructor) )
            {
//#ifdef DEBUG
              cout << "DEBUG: (Post) Invariants: first =" << d_first << endl;
//#endif /* DEBUG */

              num += addPostChecks(def, body, d_invariants);
            } 

            if (final != NULL)
            {
              numPrep[1] += addFinalize(def, body, final);
              num += 1;
            }
          } /* end if have an annotation destination */
        } /* end if have annotations to make */

        clauses.clear();

#ifdef DEBUG
        cout<<"\nDEBUG:BEGIN **********************************\n";
        cout<<"Instrumented routine:  "<<nm.getString()<<"():\n";
        cout<<"  Checks (includes non-executable)\n";
        cout<<"    Preconditions  = "<<numChecks[0]<<"\n";
        cout<<"    PostConditions = "<<numChecks[1]<<"\n";
        cout<<"    Invariants     = "<<numChecks[2]<<"\n";
        cout<<"  Prep Statements\n";
        cout<<"    Initialization = "<<numPrep[0]<<"\n";
        cout<<"    Finalization   = "<<numPrep[1]<<"\n";
        cout<<"  Total Statements = "<<num<<"\n";
        cout<<"DEBUG:END ************************************\n\n";
#endif /* DEBUG */
      } /* end if comments */
    } /* end if have declaration */
  } /* end if have definition */

  return num;
}  /* processComments */


/**
 * Process any contract annotations associated with the function declaration 
 * node.
 *
 * @param[in] def  Function definition node.
 * @return         The number of statements added.
 */
int
ContractsProcessor::processFunctionDef(
  /* inout */ SgFunctionDefinition* def)
{
  int num = 0;

  if (def != NULL)
  {
    SgFunctionDeclaration* decl = def->get_declaration();
    if (decl != NULL)
    {
      SgFunctionDeclaration* defDecl =
        isSgFunctionDeclaration(decl->get_definingDeclaration());
      if ( (defDecl != NULL) && (defDecl == decl) )
      {
#ifdef DEBUG
        printLineComment(def, "DEBUG: Have function declaration node.", false);
#endif /* DEBUG */

        d_first = true;
        num = processFunctionComments(def);
      }
    }
  }

  return num;
}  /* processFunctionDef */


/**
 * Process any contract annotations associated with a general (i.e.,
 * non-function) node.
 *
 * @param lNode   [inout] Current AST (located) node.
 * @return        The number of statements added.
 *
 * @warning  The work-around for adding embedded contract initialization and
 * finalization calls assumes we are only generating checks in C/C++.
 */
int
ContractsProcessor::processNonFunctionNode(
  /* inout */ SgLocatedNode* lNode)
{
  int num = 0;

  if (lNode != NULL)
  {
#ifdef DEBUG
    printLineComment(lNode, "DEBUG: ..processing non-function node", false);
    cout << "Node type: " << lNode->variantT() << "(";
    cout << Cxx_GrammarTerminalNames[lNode->variantT()].name << ")\n";
#endif /* DEBUG */

    ContractClauseType clauses;
    extractContract(lNode, true, clauses);
    if (clauses.size() > 0)
    {
//#ifdef DEBUG
      cout<<"DEBUG: ..processing non-function node contract clauses\n";
//#endif /* DEBUG */

      ContractClauseType::iterator iter;
      for (iter = clauses.begin(); iter != clauses.end(); iter++)
      {
        ContractComment* cc = (*iter);
        if (cc != NULL)
        {
          switch (cc->type())
          {
          case ContractComment_PRECONDITION:
            {
              printLineComment(lNode, S_ERROR_PRECONDITIONS, false);
            }
            break;
          case ContractComment_POSTCONDITION:
            {
              printLineComment(lNode, S_ERROR_POSTCONDITIONS, false);
            }
            break;
          case ContractComment_INVARIANT:
            {
              setInvariants(lNode, cc);
            }
            break;
          case ContractComment_INIT:
            {
              num += processInit(lNode, cc);
            }
            break;
          case ContractComment_FINAL:
            {
              num += processFinal(lNode, cc);
            }
            break;
          case ContractComment_ASSERT:
            {
              num += processAssert(lNode, cc);
            }
            break;
          case ContractComment_NONE:
            {
//#ifdef DEBUG
              cout << "DEBUG: ....No contract comment(?)\n";
//#endif /* DEBUG */
            }
            break;
          default:
            {
//#ifdef DEBUG
              cout << "DEBUG: ....unrecognized contract comment\n";
//#endif /* DEBUG */
            }
            break;
          } /* end switch */
        } /* end if have contract comment to process */
      } /* end for each comment */

      clauses.clear();

      if (num > 1)
      {
        printLineComment(lNode, S_WARN_MULTI_NOFUNC, false);
      }

      d_first = false;
    } /* end if have comments */
  } /* end if have a node */

  return num;
}  /* processNonFunctionNode */


/**
 * Sets the specified (invariant) contract comment to the specified AST 
 * node IF it is not already set.
 *
 * @param lNode   [inout] Current AST (located) node.
 * @param[in] cc  Invariant contract comment.
 */
void 
ContractsProcessor::setInvariants(SgLocatedNode* lNode, ContractComment* cc) {
#ifdef DEBUG
  cout<<"DEBUG: ....processing INVARIANT clause\n";
#endif /* DEBUG */

  if ( (lNode != NULL) && (cc != NULL) && (cc->isInvariant()) ) {
    if (d_invariants == NULL)
    {
      d_invariants = cc;

#if 0
      /* 
       * The following adds a C comment to .cc files BUT simple
       * text to .cpp files.
       *
       */
      SgStatement* currSttmt = isSgStatement(lNode);
      if (currSttmt != NULL)
      {
        SageInterface::addMessageStatement(currSttmt, S_NOTE_INVARIANTS);

       // Ideally would use the following instead of adding simple :
        SageInterface::attachComment(currSttmt, S_NOTE_INVARIANTS,
          PreprocessingInfo::before, cc->directive());
      }
#endif

      SageInterface::addTextForUnparser(lNode, S_NOTE_INVARIANTS,
        AstUnparseAttribute::e_before);
    }
    else
    {
      printLineComment(lNode, S_WARN_INVARIANTS, false);
    }
  }

  return;
}  /* setInvariants */


/**
 * Process an ASSERT clause associated with the specified node.
 *
 * @param     lNode  [inout] Current AST (located) node.
 * @param[in] cc     INIT contract comment.
 * @return           Returns a count of the number of instrumentations made 
 *                     (i.e., 1 if added, 0 otherwise).
 */
int
ContractsProcessor::processAssert(SgLocatedNode* lNode, ContractComment* cc)
{
  int num = 0;

#ifdef DEBUG
  cout<<"DEBUG: ....processing ASSERT clause\n";
#endif /* DEBUG */

  if ( (lNode != NULL) && (cc != NULL) && cc->isAssert() ) 
  {
    SgStatement* currSttmt = isSgStatement(lNode);
    if (currSttmt != NULL)
    {
      ContractClauseEnum ccType = cc->clause();

      list<AssertionExpression> aeList = cc->getList();
      for(list<AssertionExpression>::iterator iter = aeList.begin();
          iter != aeList.end(); iter++)
      {
        AssertionExpression ae = (*iter);
        switch (ae.support())
        {
        case AssertionSupport_EXECUTABLE:
          {
            SgExprStatement* sttmt = buildCheck(currSttmt, ccType, ae, 
                                                cc->directive());
            if (sttmt != NULL)
            {
              SageInterface::insertStatementBefore(currSttmt, sttmt, true);
              num+=1;
            }
          }
          break;
        case AssertionSupport_UNSUPPORTED:
          {
            SageInterface::attachComment(currSttmt, 
              S_PREFACE + string(S_CONTRACT_CLAUSE[ccType])
                + ": " + L_UNSUPPORTED_EXPRESSION + ae.expr(),
              PreprocessingInfo::after, cc->directive());
          }
          break;
        default:
          {
            /* Nothing to do here */
          }
          break;
        }
      }
    }
    else
    {
      printLineComment(lNode, S_ERROR_NOT_SCOPED, false);
    }
  }

  return num;
} /* processAssert */


/**
 * Process a FINAL clause associated with the specified node.
 *
 * @param     lNode  [inout] Current AST (located) node.
 * @param[in] cc     Final contract comment.
 * @return           Returns a count of the number of instrumentations made 
 *                     (i.e., 1 if added, 0 otherwise).
 */
int
ContractsProcessor::processFinal(SgLocatedNode* lNode, ContractComment* cc)
{
  int num = 0;

//#ifdef DEBUG
  cout<<"DEBUG: ....processing FINAL clause\n";
//#endif /* DEBUG */

  if ( (lNode != NULL) && (cc != NULL) && cc->isFinal() ) 
  {
    SgStatement* currSttmt = isSgStatement(lNode);
    if (currSttmt != NULL)
    {
      SgExprStatement* sttmt = buildFinal(currSttmt, cc->directive());
      if (sttmt != NULL)
      {
        SageInterface::insertStatementBefore(currSttmt, sttmt,true);
        num += 1;
      }
    }
  }

  return num;
} /* processFinal */


/**
 * Process an INIT clause associated with the specified node.
 *
 * @param     lNode  [inout] Current AST (located) node.
 * @param[in] cc     INIT contract comment.
 * @return           Returns a count of the number of instrumentations made 
 *                     (i.e., 1 if added, 0 otherwise).
 */
int
ContractsProcessor::processInit(SgLocatedNode* lNode, ContractComment* cc)
{
  int num = 0;

#ifdef DEBUG
  cout<<"DEBUG: ....processing INIT clause\n";
#endif /* DEBUG */

  if ( (lNode != NULL) && (cc != NULL) && cc->isInit() ) 
  {
    SgStatement* currSttmt = isSgStatement(lNode);
    if (currSttmt != NULL)
    {
      SgExprStatement* sttmt = buildInit(currSttmt, cc->directive());
      if (sttmt != NULL)
      {
        SageInterface::insertStatementBefore(currSttmt, sttmt,true);
        num += 1;
      }
    }
  }

  return num;
} /* processInit */
