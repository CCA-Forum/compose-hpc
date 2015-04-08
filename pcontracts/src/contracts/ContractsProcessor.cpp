/**
 * \internal
 * File:           ContractsProcessor.cpp
 * Author:         T. Dahlgren
 * Created:        2012 November 1
 * Last Modified:  2015 January 13
 * \endinternal
 *
 * @file
 * @brief
 * Basic contract clause processing utilities.
 *
 * @todo At some point should be more flexible in format of INIT with 
 * optional configuration filename.
 * 
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

const string S_DUMP = "Data Dump";
const string S_FINAL = "Finalization";
const string S_INIT = "Initialization";
const string S_TIME = "Routine Time Update";
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
const string S_ERROR_BAD_CHECKS = 
  "ERROR: Refusing to add non-postcondition/invariant checks to routine end.";
const string S_ERROR_INIT = 
  "ERROR: INIT statement CANNOT have more than one filename 'expression'.";
const string S_ERROR_POSTCONDITIONS = 
  "ERROR: Postconditions MUST be associated with function definitions.";
const string S_ERROR_PRECONDITIONS = 
  "ERROR: Preconditions MUST be associated with function definitions.";
const string S_ERROR_STATS = 
  "ERROR: STATS statement CANNOT have more than one comment 'expression'.";

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

const string S_WARN_NO_EXECS = 
  "WARNING: Expecting first clause but no executable contract checks.";

const string S_SEP = ";";


/**
 * Build a contracts stats dump statement.
 *
 * @param[in] currSttmt  The current AST statement node.
 * @param[in] dt         The pre-processing directive type.
 * @param[in] desc       A _brief_ description of the purpose of the dump.
 * @return               The function call statement or NULL.
 */
SgExprStatement*
buildDump(
  /* in */ const SgStatement*  currSttmt,
  /* in */ PPIDirectiveType    dt,
  /* in */ string              desc)
{
  SgExprStatement* sttmt = NULL;

  SgExprListExp* parms = new SgExprListExp(FILE_INFO);
  if ( (currSttmt != NULL) && (parms != NULL) )
  {
    parms->append_expression(SageBuilder::buildOpaqueVarRefExp("pce_enforcer",
        currSttmt->get_scope()));
    if (desc.empty()) {
      parms->append_expression(SageBuilder::buildOpaqueVarRefExp("NULL",
          currSttmt->get_scope()));
    } else {
      parms->append_expression(SageBuilder::buildStringVal(desc));
    }
    sttmt = SageBuilder::buildFunctionCallStmt("PCE_DUMP_STATS", 
      SageBuilder::buildVoidType(), parms, currSttmt->get_scope());
    if (sttmt != NULL)
    {
      sttmt->set_parent(currSttmt->get_parent());

#ifdef PCE_ADD_COMMENTS
      SageInterface::attachComment(sttmt, S_PREFACE + S_DUMP + S_BEGIN,
        PreprocessingInfo::before, dt);
      SageInterface::attachComment(sttmt, S_PREFACE + S_DUMP + S_END,
        PreprocessingInfo::after, dt);
#endif /* PCE_ADD_COMMENTS */
#ifdef DEBUG
    }
    else
    {
        cout<<"DEBUG: ....Sage failed to build PCE_DUMP_STATS statement\n";
#endif /* DEBUG */
    }
#ifdef DEBUG
  }
  else
  {
    cout<<"DEBUG: ....buildDump not passed or unable to gen required args\n";
#endif /* DEBUG */
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
  /* in */ const SgStatement* currSttmt,
  /* in */ PPIDirectiveType   dt)
{
  SgExprStatement* sttmt = NULL;

  SgExprListExp* parms = new SgExprListExp(FILE_INFO);
  if ( (currSttmt != NULL) && (parms != NULL) )
  {
    sttmt = SageBuilder::buildFunctionCallStmt("PCE_FINALIZE", 
      SageBuilder::buildVoidType(), parms, currSttmt->get_scope());
    if (sttmt != NULL)
    {
      sttmt->set_parent(currSttmt->get_parent());

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
 * @param[in] filename   The contracts configuration filename.
 * @return               The function call statement or NULL.
 */
SgExprStatement*
buildInit(
  /* in */ const SgStatement* currSttmt,
  /* in */ PPIDirectiveType   dt,
  /* in */ string             filename)
{
  SgExprStatement* sttmt = NULL;

  SgExprListExp* parms = new SgExprListExp(FILE_INFO);
  if ( (currSttmt != NULL) && (parms != NULL) )
  {
    if (filename.empty()) {
      parms->append_expression(SageBuilder::buildOpaqueVarRefExp("NULL", 
          currSttmt->get_scope()));
    } else {
      parms->append_expression(SageBuilder::buildStringVal(filename));
    }

    sttmt = SageBuilder::buildFunctionCallStmt("PCE_INITIALIZE", 
      SageBuilder::buildVoidType(), parms, currSttmt->get_scope());
    if (sttmt != NULL)
    {
      sttmt->set_parent(currSttmt->get_parent());

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
 * Build estimated time update routine call.
 *
 * @param[in] currSttmt  The current AST statement node.
 * @return               The function call statement or NULL.
 */
SgExprStatement*
buildTimeUpdate(
  /* in */ const SgStatement* currSttmt)
{
  SgExprStatement* sttmt = NULL;

  SgExprListExp* parms = new SgExprListExp(FILE_INFO);
  if ( (currSttmt != NULL) && (parms != NULL) )
  {
    parms->append_expression(SageBuilder::buildOpaqueVarRefExp("pce_enforcer",
        currSttmt->get_scope()));
    parms->append_expression(SageBuilder::buildOpaqueVarRefExp(
      "pce_def_times.routine", currSttmt->get_scope()));
    sttmt = SageBuilder::buildFunctionCallStmt("PCE_UPDATE_EST_TIME", 
      SageBuilder::buildVoidType(), parms, currSttmt->get_scope());
    if (sttmt != NULL)
    {
      sttmt->set_parent(currSttmt->get_parent());

#ifdef PCE_ADD_COMMENTS
      SageInterface::attachComment(sttmt, S_PREFACE + S_TIME + S_BEGIN,
        PreprocessingInfo::before, dt);
      SageInterface::attachComment(sttmt, S_PREFACE + S_TIME + S_END,
        PreprocessingInfo::after, dt);
#endif /* PCE_ADD_COMMENTS */
    }
  }

  return sttmt;
} /* buildTimeUpdate */


/**
 * Extract and add assertion expressions to the contract clause.
 *
 * @param[in]      clause  The contract clause text extracted from the 
 *                           structured comment.
 * @param[in,out]  cc      The resulting contract clause/comment.
 */
void
ContractsProcessor::addExpressions(
  /* in */    const string     clause, 
  /* inout */ ContractComment* cc)
{
  if (!clause.empty() && cc != NULL)
  {
    size_t startAE = 0, endAE;
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


        /*
         * Order of expression, expr, checks IS important.  
         *
         * For example, all built-in, non-executable expressions should
         * be checks prior to invoking isExectuable().
         */
        if (expr == "is pure") 
        {
          cc->setIsPure(true);
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
        else if (cc->isInit()) // Assuming the expression is a filename
        {
          if (cc->size() <= 0)
          {
            AssertionExpression ae (label, expr, AssertionSupport_FILENAME);
            cc->add(ae);
#ifdef DEBUG
            cout << "DEBUG: ..(assuming) is an INIT filename.\n";
#endif /* DEBUG */
          } 
          else 
          {
            cerr << S_ERROR_INIT << " Ignoring any additional entries.\n";
          } 
        } 
        else if (cc->isStats()) // Assuming the expression is a brief comment
        {
          if (cc->size() <= 0)
          {
            AssertionExpression ae (label, expr, AssertionSupport_COMMENT);
            cc->add(ae);
#ifdef DEBUG
            cout << "DEBUG: ..(assuming) is an STATS comment/desc.\n";
#endif /* DEBUG */
          } 
          else 
          {
            cerr << S_ERROR_STATS << " Ignoring any additional entries.\n";
          } 
        } 
        else if (isExecutable(expr))
        {
          AssertionExpression ae (label, expr, AssertionSupport_EXECUTABLE);
          cc->add(ae);
#ifdef DEBUG
            cout << "DEBUG: ..is executable expression.\n";
#endif /* DEBUG */
        } 
        else
        {
          AssertionExpression ae (label, expr, AssertionSupport_UNSUPPORTED);
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
 * @param[in] def   Function definition.
 * @param[in] body  Function body.
 * @param[in] cc    (FINAL) Contract comment.
 * @return          Returns number of finalization calls added.    
 */
int
ContractsProcessor::addFinalize(
  /* in */    const SgFunctionDefinition* def, 
  /* inout */ const SgBasicBlock*         body, 
  /* in */    ContractComment*            cc)
{
  int num = 0;

  if ( (def != NULL) && (body != NULL) && (cc != NULL) )
  {
    PPIDirectiveType dt = cc->directive();

    SgExprStatement* sttmt = buildFinal(body, dt);
    if (sttmt != NULL)
    {
      // Call our special version of SageInterface's instrumentEndOfFunction,
      // which knows how to handle void "functions".
      num += instrumentReturnPoints(def->get_declaration(), sttmt);
    }
  }
              
  return num;
}  /* addFinalize */


/**
 * Add requisite include file(s).
 *
 * @param[in,out]  globalScope  The Sage project representing the initial AST 
 *                                of the file(s).
 * @return                      The processing status: 0 for success, non-0 
 *                                for failure.
 */
int
ContractsProcessor::addIncludes(
  /* inout */ SgGlobal* globalScope)
{
  int status = 0;

  if ( globalScope != NULL )
  {
    SageInterface::insertHeader("contracts.h", 
      PreprocessingInfo::before, false, globalScope);

    SageInterface::insertHeader("ContractsEnforcer.h", 
      PreprocessingInfo::before, false, globalScope);

    SageInterface::insertHeader("ExpressionRoutines.h", 
      PreprocessingInfo::before, false, globalScope);

    SageInterface::insertHeader("contractOptions.h", 
      PreprocessingInfo::before, false, globalScope);

    /*
     * Assuming sufficient to determine if exit (for stdlib.h) is
     * defined to know if one of the include files is present.  A
     * bit of an inefficient hack, but the check is only done once
     * per file.
     */
    string decls = globalScope->unparseToString();
    if (decls.find("void exit") == string::npos)
    {
      SageInterface::insertHeader("stdlib.h", PreprocessingInfo::before, false,
        globalScope);
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
 * @param[in,out]  project         The Sage project representing the initial 
 *                                   AST of the file(s).
 * @param[in]      skipTransforms  True if the transformations are to be 
 *                                   skipped; otherwise, false.
 * @return                         The processing status: 0 for success, non-0 
 *                                   for failure.
 */
int
ContractsProcessor::addIncludes(
  /* inout */ SgProject* project, 
  /* in */    bool       skipTransforms)
{
  int status = 0;

  if (project != NULL)
  {
    Rose_STL_Container<SgNode*> globalScopeList = 
      NodeQuery::querySubTree(project, V_SgGlobal);
    for (Rose_STL_Container<SgNode*>::iterator i = globalScopeList.begin();
         i != globalScopeList.end(); i++)
    {
      Sg_File_Info* info        = NULL;
      SgGlobal*     globalScope = NULL;
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
 * @param[in,out]  body  Pointer to the function body.
 * @param[in]      cc    (INIT) Contract comment.
 * @return               Number of initialization calls added.
 */
int
ContractsProcessor::addInitialize(
  /* inout */ SgBasicBlock*    body, 
  /* in */    ContractComment* cc)
{
  int num = 0;

  if ( (body != NULL) && (cc != NULL) )
  {
#ifdef DEBUG
      cout<<"DEBUG: ....INIT filename =\""<<cc->getFilename()<<"\"\n";
#endif /* DEBUG */
    SgExprStatement* sttmt = buildInit(body, cc->directive(), 
                                       cc->getFilename());
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
 * @param[in]      def   Function definition.
 * @param[in,out]  body  Pointer to the function body, which is assumed
 *                         to belong to the function definition, def.
 * @param[in]      cc    The contract comment whose expressions are to be added.
 * @return               The number of statements added to the body.
 */
int
ContractsProcessor::addPostChecks(
  /* in */    const SgFunctionDefinition* def, 
  /* inout */ SgBasicBlock*               body, 
  /* in */    ContractComment*            cc)
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
                // Call our special version of SageInterface's 
                // instrumentEndOfFunction, which knows how to handle 
                // returns without values in void "functions".
                num += instrumentReturnPoints(def->get_declaration(), sttmt);
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
    } 
    else
    { 
      cerr<<"\n"<<S_ERROR_BAD_CHECKS<<"n";
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
 * @param[in,out]  body  Pointer to the function body, which is assumed to 
 *                         belong to an SgFunctionDefinition node.
 * @param[in]      cc    The contract clause/comment whose (executable) 
 *                         expressions are to be added.
 * @return               The number of statements added to the body.
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
      cerr<<"\n"<<S_ERROR_BAD_CHECKS<<"n";
    } /* end if have something to work with */

#ifdef DEBUG
    cout << "DEBUG: addPreChecks: Number of statements prepended = "<<num<<"\n";
#endif /* DEBUG */
  } /* end if have something to work with */

  return num;
}  /* addPreChecks */


/**
 * Build and add contract enforcement statistics (dump) call.
 *
 * @param[in]     def   Function definition.
 * @param[in,out] body  Function body.
 * @param[in]     cc    (STATS) Contract comment.
 * @return              Returns number of finalization calls added.    
 */
int
ContractsProcessor::addStatsDump(
  /* in */    const SgFunctionDefinition* def, 
  /* inout */ SgBasicBlock*               body, 
  /* in */    ContractComment*            cc)
{
  int num = 0;

  if ( (def != NULL) && (body != NULL) && (cc != NULL) )
  {
#ifdef DEBUG
    cout<<"DEBUG: ....STATS comment =\""<<cc->getComment()<<"\"\n";
#endif /* DEBUG */

    SgExprStatement* sttmt = buildDump(body, cc->directive(), cc->getComment());
    if (sttmt != NULL)
    {
      // Call our special version of SageInterface's instrumentEndOfFunction,
      // which knows how to handle returns without values in void "functions".
      num += instrumentReturnPoints(def->get_declaration(), sttmt);
    }
  }
              
  return num;
}  /* addStatsDump */


/**
 * Build and add the contract enforcement (routine's estimated) time update
 * call.
 *
 * @param[in,out]  body  Pointer to the function body.
 * @return               Number of initialization calls added.
 */
int
ContractsProcessor::addTimeUpdate(
  /* inout */ SgBasicBlock* body)
{
  int num = 0;

  if (body != NULL) 
  {
    SgExprStatement* sttmt = buildTimeUpdate(body);
    if (sttmt != NULL) {
      body->prepend_statement(sttmt);
      num += 1;
    }
  }
              
  return num;
}  /* addTimeUpdate */


/**
 * Build and add the return variable.
 *
 * @param[in,out]  body        Pointer to the function body.
 * @param[in]      cc          Contract comment.
 * @param[in]      returnType  The function's return type.
 *
 * @return Number of declarations (successfully) added.
 */
int
ContractsProcessor::addReturnVariable(
  /* inout */ SgBasicBlock*    body, 
  /* in */    ContractComment* cc,
  /* in */    SgType*          returnType)
{
  int num = 0;

  if ( (body != NULL) && (cc != NULL) && (returnType != NULL) )
  {
    SgVariableDeclaration* varDecl = new SgVariableDeclaration(FILE_INFO,
        "pce_result", returnType);
    if (varDecl != NULL) {
      varDecl->set_parent(body);
      body->prepend_statement(varDecl);
      num += 1;
    }
  }
              
  return num;
}  /* addReturnVariable */


/**
 * Build the contract clause check statement.
 *
 * @param[in]  currSttmt  Pointer to the current statement.
 * @param[in]  clauseType The type of contract clause associated with the 
 *                          expression.
 * @param[in]  ae         The assertion expression.
 * @param[in]  dt         The (comment) directive type.
 * @return                Contract clause statement node.
 */
SgExprStatement*
ContractsProcessor::buildCheck(
  /* in */ const SgStatement*  currSttmt, 
  /* in */ ContractClauseEnum  clauseType, 
  /* in */ AssertionExpression ae, 
  /* in */ PPIDirectiveType    dt)
{
  SgExprStatement* sttmt = NULL;

  if ( (currSttmt != NULL) && (ae.support() == AssertionSupport_EXECUTABLE) )
  {
    string cmt, clauseTypeStr, clauseTime;
    SgExprListExp* parms = new SgExprListExp(FILE_INFO);

    if (parms != NULL)
    {
      string cmt, clauseTypeStr, clauseTime;
      parms->append_expression(SageBuilder::buildOpaqueVarRefExp(
          "pce_enforcer", currSttmt->get_scope()));
  
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

      parms->append_expression(SageBuilder::buildOpaqueVarRefExp(clauseTypeStr,
          currSttmt->get_scope()));
      parms->append_expression(SageBuilder::buildOpaqueVarRefExp(clauseTime,
          currSttmt->get_scope()));
      parms->append_expression(SageBuilder::buildOpaqueVarRefExp(
        "pce_def_times.routine", currSttmt->get_scope()));
      parms->append_expression(new SgStringVal(FILE_INFO, ae.label()));
      parms->append_expression(SageBuilder::buildOpaqueVarRefExp(
          "("+ae.expr()+")", currSttmt->get_scope()));
  
      sttmt = SageBuilder::buildFunctionCallStmt("PCE_CHECK_EXPR_TERM", 
        SageBuilder::buildVoidType(), parms, currSttmt->get_scope());

#ifdef PCE_ADD_COMMENTS
      if (sttmt != NULL)
      {
        sttmt->set_parent(currSttmt->get_parent());

        SageInterface::attachComment(sttmt, S_PREFACE + cmt + S_BEGIN,
          PreprocessingInfo::before, dt);
        SageInterface::attachComment(sttmt, S_PREFACE + cmt + S_END,
          PreprocessingInfo::after, dt);
      }
#ifdef DEBUG
      else
      {
        cout << "DEBUG: buildCheck: New statement is NULL.\n";
      }
#endif /* DEBUG */
#endif /* PCE_ADD_COMMENTS */
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
 * @param[in]      lNode    Current located AST node.
 * @param[in,out]  clauses  The contract, which may consist of one or more 
 *                            clauses.
 */
void
ContractsProcessor::extractContract(
  /* in */    SgLocatedNode*     lNode,
  /* inout */ ContractClauseType &clauses)
{
  if (lNode != NULL)
  {
    AttachedPreprocessingInfoType* cmts = lNode->getAttachedPreprocessingInfo();
    if (cmts != NULL)
    {
      int numComments = 0;

      AttachedPreprocessingInfoType::iterator iter;
      for (iter = cmts->begin(); iter != cmts->end(); iter++)
      {
        /* Delete any comments if we've hit the end of a macro */
        PreprocessingInfo::DirectiveType dt = (*iter)->getTypeOfDirective();
        if (dt == PreprocessingInfo::CpreprocessorEndifDeclaration) 
        {
#ifdef DEBUG
            cout << "DEBUG: ....Encountered endif, removing " << clauses.size() <<" comments\n";
#endif /* DEBUG */
          while (!clauses.empty()) 
          {
            numComments--;
            clauses.pop_back();
          }
        }
        else 
        {
          ContractComment* cc = extractContractComment(lNode, iter);
          if (cc != NULL)
          {
            numComments++;

#ifdef DEBUG
            if (numComments <= 1) 
            {
              printLineComment(lNode, "DEBUG: ..Processing..", false);
            }
            cout << "DEBUG: ....Pushing back node clause #" << numComments <<"\n";
#endif /* DEBUG */
            clauses.push_back(cc);
          } /* end if have contract comment to process */
        }
      } /* end for each comment */

      if (numComments != clauses.size())
      {
        cerr << "ERROR: Expected " << numComments << " contract clauses but ";
        cerr << "only " << clauses.size() << " saved off.\n";
      }
    } /* end if have comments */
  } /* end if have a node */

  return;
}  /* extractContract */


/**
 * Extract the contract clause comment, if any, from the pre-processing 
 * directive.
 *
 * @param[in]  aNode  Current AST node.
 * @param[in]  info   The preprocessing directive.
 * @return           The ContractComment type.
 */
ContractComment*
ContractsProcessor::extractContractComment(
  /* in */ SgNode*                                 aNode, 
  /* in */ AttachedPreprocessingInfoType::iterator info)
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
          cc = processCommentEntry(aNode, str.substr(2, str.size()-4), dt);
        }
        break;
      case PreprocessingInfo::CplusplusStyleComment:
        {
          string str = (*info)->getString();
          cc = processCommentEntry(aNode, str.substr(2), dt);
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
 * Determine if the located node has an associated contract comment.
 *
 * @param[in] lNode  Current located AST node.
 *
 * @return True if lNode has an associated contract comment; false otherwise.
 */
bool
ContractsProcessor::hasContractComment(
  /* in */ SgLocatedNode* lNode)
{
  bool hasClause = false;
  
  AttachedPreprocessingInfoType* cmts = lNode->getAttachedPreprocessingInfo();
  if (cmts != NULL)
  {
    AttachedPreprocessingInfoType::iterator iter;
    for (iter = cmts->begin(); iter != cmts->end() && !hasClause; iter++)
    {
      if (isCComment(((*iter)->getTypeOfDirective())))
      {
        string str = (*iter)->getString();
        if (str.find("\%CONTRACT")!=string::npos)
        {
          hasClause = true;
        }
      }
    }
  }

  return hasClause;
} /* hasContractComment */


/**
 * Determine if what is assumed to be the method name is in the 
 * specified clause.
 *
 * @param[in]  nm  Method name.
 * @param[in]  cc  Contract clause.
 * @return         True if nm is in at least one invariant expression; false 
 *                   otherwise.
 */
bool
ContractsProcessor::inClause(
  /* in */ string nm,
  /* in */ ContractComment* cc)
{
  bool isIn = false;
  
  if ( !nm.empty() && (cc != NULL) )
  {
    list<AssertionExpression> aeList = cc->getList();
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
}  /* inClause */


/**
 * Add (test) contract assertion checks to each routine.
 *
 * @param[in,out]  project         The Sage project representing the initial
 *                                   AST of the file(s).
 * @param[in]      skipTransforms  True if the transformations are to be 
 *                                   skipped; otherwise, false.
 * @return                         The processing status: 0 for success, non-0 
 *                                   for failure.
 */
int
ContractsProcessor::instrumentRoutines(
  /* inout */ SgProject* project, 
  /* in */    bool       skipTransforms)
{
  int status = 0;
  if (project != NULL)
  {
#if 0
    /* Run internal consistency checks on the AST _before_ changing it. */
#ifdef DEBUG
    cout<<"DEBUG: Checking internal consistency of the AST before changes...\n";
#endif /* DEBUG */
    AstTests::runAllTests(project);
#endif /* 0 */

    /* Find all function definitions. */
#ifdef DEBUG
    cout<<"DEBUG: Calling querySubTree...\n";
#endif /* DEBUG */

    vector<SgNode*> fdList = 
      NodeQuery::querySubTree(project, V_SgFunctionDefinition);

    if (!fdList.empty())
    {
      int num = 0;

      vector<SgNode*>::iterator iter;
      for (iter=fdList.begin(); iter!=fdList.end(); iter++)
      {
        Sg_File_Info* info = NULL;
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
#ifdef DEBUG
            cout<<"DEBUG: Calling processFunctionDef...\n";
#endif /* DEBUG */
            int chgd = processFunctionDef(def);
            num += chgd;

#ifdef DEBUG
            cout<<"DEBUG: Calling fixVariableReferences...\n";
#endif /* DEBUG */
            int fixed = SageInterface::fixVariableReferences(*iter);
            cout<<"Fixed "<<fixed<<" variable refs after "<<chgd<<" changes.\n";
          }
        }
      }

      cout<<"Added "<<num<<" contract-related statements.\n";
    }

    /* Translate the file(s) */
#if 0
#ifdef DEBUG
    cout<<"DEBUG: Calling unparse...\n";
#endif /* DEBUG */
    project->unparse();
#endif

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
 * @param[in]  expr  The string representing the assertion expression.
 * @return           True if the expression appears to be executable; False 
 *                     otherwise.
 */
bool
ContractsProcessor::isExecutable(
  /* in */ string expr)
{
  bool isOkay = true;

  if (!expr.empty())
  {
    for (int i=MIN_NEE_INDEX; i<MAX_NEE_INDEX; i++)
    {
      if (expr.find(UnsupportedInterfaces[i]) != string::npos)
      {
#ifdef DEBUG
        cout << "DEBUG: Detected \'" << UnsupportedInterfaces[i];
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
 * @param[in,out]  aNode   Current AST node.
 * @param[in]      cmt     Comment contents.
 * @param[in]      dirType (Comment) directive type.
 * @return                 The corresponding ContractComment type.
 */
ContractComment*
ContractsProcessor::processCommentEntry(
  /* inout */ SgNode*                          aNode, 
  /* in */    string                           cmt, 
  /* in */    PreprocessingInfo::DirectiveType dirType)
{
  ContractComment* cc = NULL;

  if ( (aNode != NULL) && !cmt.empty() )
  {
    size_t pos;
    if ((pos=cmt.find("\%CONTRACT"))!=string::npos)
    {
      if ((pos=cmt.find(" REQUIRE"))!=string::npos)
      {
        cc = new ContractComment(ContractComment_PRECONDITION, dirType);
        addExpressions(cmt.substr(pos+8), cc);
#ifdef DEBUG
        cout<<"DEBUG: Created REQUIRE ContractComment: "<<cc->str(S_SEP)<<endl;
#endif /* DEBUG */
      }
      else if ((pos=cmt.find(" ENSURE"))!=string::npos)
      {
        cc = new ContractComment(ContractComment_POSTCONDITION, dirType);
        addExpressions(cmt.substr(pos+7), cc);
#ifdef DEBUG
        cout<<"DEBUG: Created ENSURE ContractComment: "<<cc->str(S_SEP)<<endl;
#endif /* DEBUG */
      }
      else if ((pos=cmt.find(" INVARIANT"))!=string::npos)
      {
        cc = new ContractComment(ContractComment_INVARIANT, dirType);
        addExpressions(cmt.substr(pos+10), cc);
#ifdef DEBUG
        cout<<"DEBUG: Created INVARIANT ContractComment: ";
        cout<<cc->str(S_SEP)<<endl;
#endif /* DEBUG */
      }
      else if ((pos=cmt.find(" ASSERT"))!=string::npos)
      {
        cc = new ContractComment(ContractComment_ASSERT, dirType);
        addExpressions(cmt.substr(pos+7), cc);
#ifdef DEBUG
        cout<<"DEBUG: Created ASSERT ContractComment: ";
        cout<<cc->str(S_SEP)<<endl;
#endif /* DEBUG */
      }
      else if ((pos=cmt.find(" INIT"))!=string::npos)
      {
        cc = new ContractComment(ContractComment_INIT, dirType);
        addExpressions(cmt.substr(pos+6), cc);
#ifdef DEBUG
        cout<<"DEBUG: Created INIT ContractComment: ";
        cout<<cc->str(S_SEP)<<endl;
#endif /* DEBUG */
      }
      else if ((pos=cmt.find(" FINAL"))!=string::npos)
      {
        cc = new ContractComment(ContractComment_FINAL, dirType);
#ifdef DEBUG
        cout<<"DEBUG: Created FINAL ContractComment: "<<cc->str(S_SEP)<<endl;
#endif /* DEBUG */
      }
      else if ((pos=cmt.find(" STATS"))!=string::npos)
      {
        cc = new ContractComment(ContractComment_STATS, dirType);
        addExpressions(cmt.substr(pos+7), cc);
#ifdef DEBUG
        cout<<"DEBUG: Created STATS ContractComment: "<<cc->str(S_SEP)<<endl;
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
 * @param[in,out]  def  The function definition node.
 * @return              The number of statements added.
 */
int
ContractsProcessor::processFunctionComments(
  /* inout */ SgFunctionDefinition* def)
{
  int num = 0;

  if (def != NULL)
  {
#ifdef DEBUG
    cout<<"DEBUG: ..Calling get_declaration...\n";
#endif /* DEBUG */

    SgFunctionDeclaration* decl = def->get_declaration();
    if (decl != NULL)
    {
#ifdef DEBUG
      cout<<"DEBUG: ..obtained function declaration\n";
#endif /* DEBUG */
      bool isConstructor = false;
      bool isDestructor = false;
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
      extractContract(decl, clauses);

#ifdef DEBUG
      cout<<"DEBUG: ....extracted "<<clauses.size()<<" contract clause(s).\n";
#endif /* DEBUG */

      if (clauses.size() > 0)
      {
        bool isInitRoutine = false;
        bool isPureRoutine = false;

        ContractComment* final = NULL;
        ContractComment* init = NULL;
        ContractComment* pre = NULL;
        ContractComment* post = NULL;
        ContractComment* stats = NULL;
        int numChecks[] = { 0, 0, 0 };
        int numPrep[] = { 0, 0, 0, 0 };
        int numExec = 0;

        ContractClauseType::iterator iter;
        for (iter = clauses.begin(); iter != clauses.end(); iter++)
        {
          ContractComment* cc = (*iter);
          if (cc != NULL)
          {
            if (cc->isInInit()) { isInitRoutine = true; }
            if (cc->isPure()) { isPureRoutine = true; }

#ifdef PCE_ENABLE_WARNING
            if (cc->numExecutable() <= 0)
            {
              printLineComment(def, S_WARN_NO_EXECS, false);
            }
#endif /* PCE_ENABLE_WARNING */

            switch (cc->type())
            {
            case ContractComment_PRECONDITION:
              {
                pre = cc;
                numChecks[0] += pre->size();

                /**
                 * @todo Should executable preconditions be factored in
                 * here (instead of where they are instrumented)?
                 *
                numExec += pre->numExecutable();
                 *
                 */
              }
              break;
            case ContractComment_POSTCONDITION:
              {
                post = cc;
                numChecks[1] += post->size();

                /**
                 * @todo Should executable preconditions be factored in
                 * here (instead of where they are instrumented)?
                 *
                numExec += post->numExecutable();
                 *
                 */
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
            case ContractComment_STATS:
              {
                stats = cc;
              }
              break;
            case ContractComment_ASSERT:
              {
#ifdef DEBUG
                cout<<"DEBUG: ......erroneous ASSERT clause\n";
#endif /* DEBUG */

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
           || (post != NULL) || (stats != NULL) || (d_invariants != NULL) )
        {
          SgBasicBlock* body = def->get_body();
          bool returnAdded = false;
  
          if (body != NULL)
          {
            if (d_invariants != NULL)
            {
              numChecks[2] = d_invariants->size();

              /**
               * @todo If routine time should be factored in for main, 
               * constructor(s), non-member methods, etc., then the
               * number of executable invariants should be calculated here
               * instead of when the check is added to the code.
               * 
              numExec += 2*d_invariants->numExecutable();
               * 
               */ 
            }

            /*
             * First add initial routine instrumentation.
             * ..Order IS important since each is prepended to the body.
             */
            bool skipInvariants = (nm=="main") || (init!=NULL) || (final!=NULL)
              || (d_invariants==NULL) || (numChecks[2]<=0) || isConstructor 
              || (!isMemberFunc) || inClause(nm, d_invariants) || isPureRoutine;


            if (! (skipInvariants || isInitRoutine) )
            {
              num += addPreChecks(body, d_invariants);
              numExec += d_invariants->numExecutable();
            }

            if ( (pre != NULL) && (numChecks[0] > 0) && !inClause(nm, pre) )
            { 
              num += addPreChecks(body, pre);
              numExec += pre->numExecutable();

              if (pre->needsResult() && !returnAdded) {
                if (addReturnVariable(body, pre, decl->get_orig_return_type()))
                {
                  returnAdded = true;
                  num += 1;
                }
              }
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
              if ( (numChecks[1] > 0) && !inClause(nm, post) )
              {
                if (post->needsResult() && !returnAdded) {
                  if (addReturnVariable(body, post, 
                                        decl->get_orig_return_type()))
                  {
                    returnAdded = true;
                    num += 1;
                  }
                }
                num += addPostChecks(def, body, post);
                numExec += post->numExecutable();
              }
            }

            if (! (skipInvariants || isDestructor) )
            {
              num += addPostChecks(def, body, d_invariants);
              numExec += d_invariants->numExecutable();
            } 

            if (stats != NULL)
            {
              numPrep[2] += addStatsDump(def, body, stats);
              num += 1;
            }

            if (final != NULL)
            {
              numPrep[1] += addFinalize(def, body, final);
              num += 1;
            }

            /*
             * If ANY executable checks were added, then the routine time
             * estimate must be added FIRST to aid partial enforcement 
             * strategies.
             */
            if (numExec > 0)
            {
              d_first = false;
              numPrep[3] += addTimeUpdate(body);
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
        cout<<"    Stats dump     = "<<numPrep[2]<<"\n";
        cout<<"    Timing         = "<<numPrep[3]<<"\n";
        cout<<"  Total Statements = "<<num<<"\n";
        cout<<"        Executable = "<<numExec<<"\n";
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
 * @param[in,out]  def  Function definition node.
 * @return              The number of statements added.
 */
int
ContractsProcessor::processFunctionDef(
  /* inout */ SgFunctionDefinition* def)
{
  int num = 0;

  if (def != NULL)
  {
    /* This will be the first pass for the function. */
    d_first = true;

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
 * @param[in,out]  lNode  Current AST (located) node.
 * @return                The number of statements added.
 *
 * @warning  The work-around for adding embedded contract initialization and
 * finalization calls assumes we are only generating checks in C/C++.
 *
 * @todo Need to determine WHY this routine is apparently not called for the 
 * return statement in main of helloworld-v2.cc, which is where the 
 * test case's FINAL annotation currently resides.
 */
int
ContractsProcessor::processNonFunctionNode(
  /* inout */ SgLocatedNode* lNode)
{
  int num = 0;

  if (lNode != NULL)
  {
#ifdef DEBUG
    printLineComment(lNode, "DEBUG: ..processing non-function node..", true);
#endif /* DEBUG */

    ContractClauseType clauses;
    extractContract(lNode, clauses);
    if (clauses.size() > 0)
    {
#ifdef DEBUG
      cout<<"DEBUG: ..processing " << clauses.size() << " contract clauses\n";
#endif /* DEBUG */

      ContractClauseType::iterator iter;
      for (iter = clauses.begin(); iter != clauses.end(); iter++)
      {
        ContractComment* cc = (*iter);
        if (cc != NULL)
        {
#ifdef DEBUG
          cout << "DEBUG: ..cc=" << cc->str(",") << endl;
#endif /* DEBUG */
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
          case ContractComment_STATS:
            {
              num += processStats(lNode, cc);
            }
            break;
          case ContractComment_NONE:
            {
#ifdef DEBUG
              cout << "DEBUG: ....No contract comment(?)\n";
#endif /* DEBUG */
            }
            break;
          default:
            {
#ifdef DEBUG
              cout << "DEBUG: ....unrecognized contract comment\n";
#endif /* DEBUG */
            }
            break;
          } /* end switch */
        } /* end if have contract comment to process */
      } /* end for each comment */

      clauses.clear();
    } /* end if have comments */
  } /* end if have a node */

  return num;
}  /* processNonFunctionNode */


/**
 * Sets the specified (invariant) contract comment to the specified AST 
 * node IF it is not already set.
 *
 * @param[in,out]  lNode  Current AST (located) node.
 * @param[in]      cc     Invariant contract comment.
 */
void 
ContractsProcessor::setInvariants(
  /* inout */ SgLocatedNode*   lNode, 
  /* in */    ContractComment* cc) 
{
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

       // Ideally would use the following instead of adding simple text:
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
 * @param[in,out]  lNode  Current AST (located) node.
 * @param[in]      cc     INIT contract comment.
 * @return                Returns a count of the number of instrumentations 
 *                          made (i.e., 1 if added, 0 otherwise).
 */
int
ContractsProcessor::processAssert(
  /* inout */ SgLocatedNode*   lNode, 
  /* in */    ContractComment* cc)
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

      if ( (cc->numExecutable() > 0) && d_first )
      {
        SgExprStatement* sttmt = buildTimeUpdate(currSttmt);
        if (sttmt != NULL)
        {
          SageInterface::insertStatementBefore(currSttmt, sttmt, true);
          num+=1;
        }
      }

      list<AssertionExpression> aeList = cc->getList();
      for(list<AssertionExpression>::reverse_iterator iter = aeList.rbegin();
          iter != aeList.rend(); iter++)
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
 * Process a FINAL associated with the specified node.
 *
 * @param[in,out]  lNode  Current AST (located) node.
 * @param[in]      cc     Final contract comment.
 * @return                Returns a count of the number of instrumentations 
 *                          made (i.e., 1 if added, 0 otherwise).
 */
int
ContractsProcessor::processFinal(
  /* inout */ SgLocatedNode*   lNode, 
  /* in */    ContractComment* cc)
{
  int num = 0;

#ifdef DEBUG
  cout<<"DEBUG: ....processing FINAL\n";
#endif /* DEBUG */

  if ( (lNode != NULL) && (cc != NULL) && cc->isFinal() ) 
  {
    SgStatement* currSttmt = isSgStatement(lNode);
    if (currSttmt != NULL)
    {
      SgExprStatement* sttmt = buildFinal(currSttmt, cc->directive());
      if (sttmt != NULL)
      {
        SageInterface::insertStatementBefore(currSttmt, sttmt, true);
        num += 1;
      }
    }
  }

  return num;
} /* processFinal */


/**
 * Process an INIT associated with the specified node.
 *
 * @param[in,out]  lNode  Current AST (located) node.
 * @param[in]      cc     INIT contract comment.
 * @return                Returns a count of the number of instrumentations
 *                          made (i.e., 1 if added, 0 otherwise).
 */
int
ContractsProcessor::processInit(
  /* inout */ SgLocatedNode*   lNode, 
  /* in */    ContractComment* cc)
{
  int num = 0;

#ifdef DEBUG
  cout<<"DEBUG: ....processing INIT\n";
#endif /* DEBUG */

  if ( (lNode != NULL) && (cc != NULL) && cc->isInit() ) 
  {
    SgStatement* currSttmt = isSgStatement(lNode);
    if (currSttmt != NULL)
    {
#ifdef DEBUG
      cout<<"DEBUG: ....INIT filename =\""<<cc->getFilename()<<"\"\n";
#endif /* DEBUG */
      SgExprStatement* sttmt = buildInit(currSttmt, cc->directive(), 
                                         cc->getFilename());
      if (sttmt != NULL)
      {
        SageInterface::insertStatementBefore(currSttmt, sttmt, true);
        num += 1;
      }
    }
  }

  return num;
} /* processInit */


/**
 * Process a STATS associated with the specified node.
 *
 * @param[in,out]  lNode  Current AST (located) node.
 * @param[in]      cc     Stats contract comment.
 * @return                Returns a count of the number of instrumentations 
 *                          made (i.e., 1 if added, 0 otherwise).
 */
int
ContractsProcessor::processStats(
  /* inout */ SgLocatedNode*   lNode, 
  /* in */    ContractComment* cc)
{
  int num = 0;

#ifdef DEBUG
  cout << "DEBUG: ....processing STATS\n";
#endif /* DEBUG */

  if ( (lNode != NULL) && (cc != NULL) && cc->isStats() ) 
  {
    SgStatement* currSttmt = isSgStatement(lNode);
    if (currSttmt != NULL)
    {
      SgExprStatement* sttmt = buildDump(currSttmt, cc->directive(),
                                         cc->getComment());
      if (sttmt != NULL)
      {
#ifdef DEBUG
        cout<<"DEBUG: ......inserting stats dump statement\n";
#endif /* DEBUG */
        SageInterface::insertStatementBefore(currSttmt, sttmt, true);
        num += 1;
      }
#ifdef DEBUG
      else
      {
        cout<<"DEBUG: ......failed to build dump statement.\n";
      }
#endif /* DEBUG */
    }
#ifdef DEBUG
    else
    {
      cout<<"DEBUG: ......lNode is NOT an SgStatement so skipping.\n";
    }
#endif /* DEBUG */
  }

  return num;
} /* processStats */
