/**
 * \internal
 * File:           ContractsProcessor.cpp
 * Author:         T. Dahlgren
 * Created:        2012 November 1
 * Last Modified:  2013 February 21
 * \endinternal
 *
 * @file
 * @brief
 * Basic contract clause processing utilities.
 *
 * @htmlinclude copyright.html
 */

#include <iostream>
#include <list>
#include <string>
#include "rose.h"
#include "Cxx_Grammar.h"
#include "RoseHelpers.hpp"
#include "contractOptions.h"
#include "contractClauseTypes.hpp"
#include "ContractsProcessor.hpp"

using namespace std;

const string S_COMMENT_DUMP_BEGIN = 
  "PContract: Enforcement Data Dump: BEGIN";
const string S_COMMENT_DUMP_END = 
  "PContract: Enforcement Data Dump: END";

const string S_COMMENT_FINAL_BEGIN = 
  "PContract: Enforcement Finalization: BEGIN";
const string S_COMMENT_FINAL_END = 
  "PContract: Enforcement Finalization: END";

const string S_COMMENT_INCLUDES = 
  "PContract: Contract enforcement includes";

const string S_COMMENT_INIT_BEGIN = 
  "PContract: Enforcement Initialization: BEGIN";
const string S_COMMENT_INIT_END = 
  "PContract: Enforcement Initialization: END";

const string S_ERROR_NOT_SCOPED = 
  "ERROR: The contract annotation MUST be associated with a scoped statement.";

const string S_ERROR_PRECONDITIONS = 
  "ERROR: Preconditions MUST be associated with function definitions.";
const string S_ERROR_POSTCONDITIONS = 
  "ERROR: Postconditions MUST be associated with function definitions.";

const string S_WARN_INVARIANTS = 
  "WARNING: Ignoring additional invariant clause.";

const string S_WARN_MULTI_NOFUNC = 
  "WARNING: Multiple annotations detected for non-function node.";

const string S_WARN_NO_EXECS = 
  "WARNING: Expecting first clause but no executable contract checks.";

const string S_SEP = ";";

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
          cc->setInit(true);
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
    PPIDirectiveType dt = cc->directive();
    int i;

    /*
     * @todo TBD/FIX: Temporarily adding statistics dump here.
     */
    SgExprListExp* parmsD = new SgExprListExp(FILE_INFO);
    if (parmsD != NULL)
    {
      parmsD->append_expression(SageBuilder::buildVarRefExp("pce_enforcer"));
      parmsD->append_expression(new SgStringVal(FILE_INFO, "End processing"));
      SgExprStatement* sttmt = SageBuilder::buildFunctionCallStmt(
        "PCE_DUMP_STATS", SageBuilder::buildVoidType(), parmsD, body);
      if (sttmt != NULL)
      {
        SageInterface::attachComment(sttmt, S_COMMENT_DUMP_BEGIN,
          PreprocessingInfo::before, dt); 
        SageInterface::attachComment(sttmt, S_COMMENT_DUMP_END,
          PreprocessingInfo::after, dt);

        i = SageInterface::instrumentEndOfFunction(def->get_declaration(), 
              sttmt);
        num+=i;
      }
    }

    SgExprListExp* parmsF = new SgExprListExp(FILE_INFO);
    if (parmsF != NULL)
    {
      SgExprStatement* sttmt = SageBuilder::buildFunctionCallStmt(
        "PCE_FINALIZE", SageBuilder::buildVoidType(), parmsF, body);
      if (sttmt != NULL)
      {
        SageInterface::attachComment(sttmt, S_COMMENT_FINAL_BEGIN,
          PreprocessingInfo::before, dt);
        SageInterface::attachComment(sttmt, S_COMMENT_FINAL_END,
          PreprocessingInfo::after, dt);
        i = SageInterface::instrumentEndOfFunction(def->get_declaration(),
              sttmt);
        num+=i;
      }
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
    SageInterface::attachComment(globalScope, S_COMMENT_INCLUDES,
      PreprocessingInfo::before, PreprocessingInfo::C_StyleComment);

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
    SgExprListExp* parms = new SgExprListExp(FILE_INFO);
    if (parms != NULL)
    {
      parms->append_expression(SageBuilder::buildVarRefExp("NULL"));
      SgExprStatement* sttmt = SageBuilder::buildFunctionCallStmt(
        "PCE_INITIALIZE", SageBuilder::buildVoidType(), parms, body);
      if (sttmt != NULL)
      {
        SageInterface::attachComment(sttmt, S_COMMENT_INIT_BEGIN,
          PreprocessingInfo::before, cc->directive());
        SageInterface::attachComment(sttmt, S_COMMENT_INIT_END,
          PreprocessingInfo::after, cc->directive());
        body->prepend_statement(sttmt);
        num += 1;
      }
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
                int i = SageInterface::instrumentEndOfFunction(
                         def->get_declaration(), sttmt);
                num+=i;
                d_first = false;
#ifdef DEBUG
                cout << "DEBUG: ..added to end of function: ";
                cout << sttmt->unparseToString() << "\n";
#endif /* DEBUG */
              }
            }
            break;
          case AssertionSupport_UNSUPPORTED:
            {
#ifdef DEBUG
              cout << "DEBUG: ..unsupported expression(s)\n";
#endif /* DEBUG */
              // attach unsupported comment (NOTE: ROSE seems to ignore!)
              SageInterface::attachComment(body, 
                "PContract: " + string(S_CONTRACT_CLAUSE[ccType])
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
#ifdef DEBUG
                cout << "DEBUG: ..prepended: ";
                cout << sttmt->unparseToString() << "\n";
#endif /* DEBUG */
                num++;
                d_first = false;
              }
            }
            break;
          case AssertionSupport_UNSUPPORTED:
            {
              SageInterface::attachComment(body,
                "PContract: " + string(S_CONTRACT_CLAUSE[ccType])
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
 * @param     body       [inout] Pointer to the function body.
 * @param[in] clauseType The type of contract clause associated with the 
 *                         expression.
 * @param[in] ae         The assertion expression.
 * @param[in] dt         The (comment) directive type.
 * @return               Contract clause statement node.
 */
SgExprStatement*
ContractsProcessor::buildCheck(
  /* inout */ SgBasicBlock*       body, 
  /* in */    ContractClauseEnum  clauseType, 
  /* in */    AssertionExpression ae, 
  /* in */    PPIDirectiveType    dt)
{
  SgExprStatement* sttmt = NULL;

  if ( (body != NULL) && (ae.support() == AssertionSupport_EXECUTABLE) )
  {
    string cmt, clauseTime;
    SgExprListExp* parms = new SgExprListExp(FILE_INFO);

    if (parms != NULL)
    {
      parms->append_expression(SageBuilder::buildVarRefExp("pce_enforcer"));
  
      switch (clauseType)
      {
        case ContractClause_PRECONDITION:
          {
            parms->append_expression(SageBuilder::buildVarRefExp(
              "ContractClause_PRECONDITION"));
            clauseTime = "pce_def_times.pre";
            cmt = "PContract: Precondition";
          }
          break;
        case ContractClause_POSTCONDITION:
          {
            parms->append_expression(SageBuilder::buildVarRefExp(
              "ContractClause_POSTCONDITION"));
            clauseTime = "pce_def_times.post";
            cmt = "PContract: Postcondition";
          }
          break;
        case ContractClause_INVARIANT:
          {
            parms->append_expression(SageBuilder::buildVarRefExp(
              "ContractClause_INVARIANT"));
            clauseTime = "pce_def_times.inv";
            cmt = "PContract: Invariant";
          }
          break;
        default:
          {
            /*  WARNING: This should NEVER happen. */
            parms->append_expression(SageBuilder::buildVarRefExp(
              "ContractClause_NONE"));
            clauseTime = "0";
            cmt = "PContract Error: Contact Developer(s)!";
          }
          break;
      }

      parms->append_expression(SageBuilder::buildVarRefExp(clauseTime));
      parms->append_expression(SageBuilder::buildVarRefExp(
        "pce_def_times.routine"));
      parms->append_expression(SageBuilder::buildVarRefExp(
        (ae.isFirst() && d_first) ? "CONTRACTS_TRUE" : "CONTRACTS_FALSE"));
      parms->append_expression(new SgStringVal(FILE_INFO, ae.label()));
      parms->append_expression(SageBuilder::buildVarRefExp("("+ae.expr()+")"));
  
      sttmt = SageBuilder::buildFunctionCallStmt("PCE_CHECK_EXPR_TERM", 
        SageBuilder::buildVoidType(), parms, body);
      if (sttmt != NULL)
      {
        SageInterface::attachComment(sttmt, cmt + ": BEGIN", 
          PreprocessingInfo::before, dt);
        SageInterface::attachComment(sttmt, cmt + ": END", 
          PreprocessingInfo::after, dt);
      }
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
    if (body == NULL) cout << "DEBUG: buildCheck: Given NULL body (wrong!)\n";
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
 * invariants.
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
        cout<<"DEBUG: Created INVARIANT ContractComment: "<<cc->str(S_SEP)<<endl;
#endif /* DEBUG */
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
                if (cc->isInit()) isInitRoutine = true;
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
                if (d_invariants == NULL)
                {
                  d_invariants = cc;
                }
                else
                {
                  printLineComment(decl, S_WARN_INVARIANTS, false);
                  cc = NULL;
                }
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
            case ContractComment_NONE:
            default:
              {
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
            if (pre != NULL) 
            { 
              if (numChecks[0] > 0) 
              {
                num += addPreChecks(body, pre);
              }
            }

            if (! (skipInvariants || isInitRoutine) )
            {
              num += addPreChecks(body, d_invariants);
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
        cout<<"  Checks\n";
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
 */
int
ContractsProcessor::processNonFunctionNode(
  /* inout */ SgLocatedNode* lNode)
{
  int num = 0;

  if (lNode != NULL)
  {
#ifdef DEBUG
//    printLineComment(lNode, "DEBUG: ..processing non-function node", false);
#endif /* DEBUG */

    ContractClauseType clauses;
    extractContract(lNode, true, clauses);
    if (clauses.size() > 0)
    {
#ifdef DEBUG
      cout<<"DEBUG: ....processing contract clauses\n";
#endif /* DEBUG */

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
#ifdef DEBUG
              cout<<"DEBUG: ......processing erroneous PRECONDITION clause\n";
#endif /* DEBUG */

              printLineComment(lNode, S_ERROR_PRECONDITIONS, false);
            }
            break;
          case ContractComment_POSTCONDITION:
            {
#ifdef DEBUG
              cout<<"DEBUG: ......processing erroneous POSTCONDITION clause\n";
#endif /* DEBUG */

              printLineComment(lNode, S_ERROR_POSTCONDITIONS, false);
            }
            break;
          case ContractComment_INVARIANT:
            {
#ifdef DEBUG
              cout<<"DEBUG: ......processing INVARIANT clause\n";
#endif /* DEBUG */

              if (d_invariants == NULL)
              {
                d_invariants = cc;
              }
              else
              {
                printLineComment(lNode, S_WARN_INVARIANTS, false);
              }
            }
            break;
          case ContractComment_INIT:
            {
#ifdef DEBUG
              cout<<"DEBUG: ......processing INIT clause\n";
#endif /* DEBUG */

              SgStatement* currSttmt = isSgStatement(lNode);
              if (currSttmt != NULL)
              {
                SgScopeStatement* scope = currSttmt->get_scope();
                SgExprListExp* parms = new SgExprListExp(FILE_INFO);
                if (parms != NULL)
                {
                  parms->append_expression(SageBuilder::buildVarRefExp("NULL"));
                  SgExprStatement* sttmt = SageBuilder::buildFunctionCallStmt(
                    "PCE_INITIALIZE", SageBuilder::buildVoidType(), parms, 
                    scope);
                  if (sttmt != NULL)
                  {
                    SageInterface::attachComment(sttmt, S_COMMENT_INIT_BEGIN,
                      PreprocessingInfo::before, cc->directive());
                    SageInterface::attachComment(sttmt, S_COMMENT_INIT_END,
                      PreprocessingInfo::after, cc->directive());
                    SageInterface::insertStatementBefore(currSttmt, sttmt,true);
                    num += 1;
#ifdef DEBUG
                    cout<<"DEBUG: INIT: currSttmt: "<<currSttmt->unparseToString()<<endl;
                    cout<<"DEBUG: INIT: sttmt: "<<sttmt->unparseToString()<<endl;
#endif /* DEBUG */
                  }
                }
              }
              else
              {
                printLineComment(lNode, S_ERROR_NOT_SCOPED, false);
              }
            }
            break;
          case ContractComment_FINAL:
            {
#ifdef DEBUG
              cout<<"DEBUG: ......processing FINAL clause\n";
#endif /* DEBUG */
              SgStatement* currSttmt = isSgStatement(lNode);
              if (currSttmt != NULL)
              {
                SgExprListExp* parmsF = new SgExprListExp(FILE_INFO);
                SgScopeStatement* scope = currSttmt->get_scope();
                if ( (parmsF != NULL) && (scope != NULL) )
                {
                  SgExprStatement* sttmt = SageBuilder::buildFunctionCallStmt(
                    "PCE_FINALIZE", SageBuilder::buildVoidType(), parmsF,scope);
                  if (sttmt != NULL)
                  {
                    SageInterface::attachComment(sttmt, S_COMMENT_FINAL_BEGIN,
                      PreprocessingInfo::before, cc->directive());
                    SageInterface::attachComment(sttmt, S_COMMENT_FINAL_END,
                      PreprocessingInfo::after, cc->directive());
                    SageInterface::insertStatementBefore(currSttmt, sttmt,true);
                    num += 1;
#ifdef DEBUG
                    cout<<"DEBUG: FINAL: currSttmt"<<currSttmt->unparseToString()<<endl;
                    cout<<"DEBUG: FINAL: sttmt"<<sttmt->unparseToString()<<endl;
#endif /* DEBUG */
                  }
                }
              }
            }
            break;
          case ContractComment_NONE:
          default:
            break;
          } /* end switch */
        } /* end if have contract comment to process */
      } /* end for each comment */

      clauses.clear();

      if (num > 1)
      {
        printLineComment(lNode, S_WARN_MULTI_NOFUNC, false);
      }
    } /* end if have comments */
  } /* end if have a node */

  return num;
}  /* processNonFunctionNode */
