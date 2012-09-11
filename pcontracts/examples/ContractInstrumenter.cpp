/**
 * File:           ContractInstrumenter.cpp
 * Author:         T. Dahlgren
 * Created:        2012 August 3
 * Last Modified:  2012 September 11
 *
 * @file
 * @section DESCRIPTION
 * Experimental contract enforcement instrumentation.
 *
 *
 * @todo Do NOT generate invariants in main() and/or any routine with INIT
 *  and FINAL?
 *
 * @todo Do NOT generate invariant check in routine of same name.
 *  Technically, contracts of routines used in contracts of others are not
 *  supposed to be called.
 *
 * @todo Need new annotation for checking/dumping contract check data.
 *  BUT will need to ensure properly configured to actually dump the data.
 *
 * @todo Much more thought needed to C++ annotations in terms of instances, etc.
 *
 * @todo Consider adding comment text for unsupported checks.
 *
 * @todo Need to change to a visitor to ensure acquire invariants.
 *
 * @todo Give some thought to process for obtaining execution time 
 *  estimates for contract clauses and routine.
 *
 * @todo Add support for complex (as in non-standard C) expressions.
 *
 * 
 * @section WARNING
 * This is a VERY preliminary draft. (See todo items.)
 * 
 * @section LICENSE
 * TBD
 */

#include <iostream>
#include <list>
#include <string>
#include "rose.h"
#include "Cxx_Grammar.h"
#include "RoseHelpers.hpp"
#include "contractOptions.h"

#define FILE_INFO Sg_File_Info::generateDefaultFileInfoForTransformationNode()

using namespace std;


/*
 *************************************************************************
 * Helper classes and types
 *************************************************************************
 */

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
 */
ContractClauseEnum ContractCommentClause[] = {
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
 * Known, non-executable assertion expression contents -- assuming lower case.
 */
const string NonExecExpressions[] = {
  "implies",
  "iff",
  "pce_all",
  "pce_inrange",
};
const int MIN_NEE_INDEX = 0;
const int MAX_NEE_INDEX = 4;



/**
 * Assertion expression data.
 */
class AssertionExpression 
{
  public:
    AssertionExpression(string l, string expr) : d_label(l), d_expr(expr) {}
    string label() { return d_label; }
    string expr() { return d_expr; }

  private:
    string d_label;
    string d_expr;
};  /* class AssertionExpression */


/**
 * Contract clause data.
 */
class ContractComment
{
  public:
    ContractComment(ContractCommentEnum t): 
      d_type(t), d_needsReturn(false) {}

    ~ContractComment() { d_aeList.clear(); }

    ContractCommentEnum type() { return d_type; }
    ContractClauseEnum clause() { return ContractCommentClause[d_type]; }
    void add(AssertionExpression expr) { d_aeList.push_front(expr); }
    void setResult(bool needs) { d_needsReturn = needs; }
    list<AssertionExpression> getList() { return d_aeList; }
    void clear() { d_aeList.clear(); }
    int size() { return d_aeList.size(); }

  private:
    ContractCommentEnum        d_type;
    list<AssertionExpression>  d_aeList;
    bool                       d_needsReturn;
};  /* class ContractComment */


/*
 *************************************************************************
 * Forward declarations
 *************************************************************************
 */
int
addPreChecks(SgFunctionDefinition* def, SgBasicBlock* body, ContractComment* cc,
  bool firstTime);

int
addPostChecks(SgFunctionDefinition* def, SgBasicBlock* body, 
  ContractComment* cc, bool firstTime);

void
addExpressions(string clause, ContractComment* cc);

int
addFinalize(SgFunctionDefinition* def, SgBasicBlock* body);

int
addIncludes(SgProject* project, bool skipTransforms);

int
addInitialize(SgBasicBlock* body);

SgExprStatement*
buildCheck(SgBasicBlock* body, ContractClauseEnum clauseType, bool firstTime,
  string label, string expr);

ContractComment*
extractContractClause(SgFunctionDeclaration* decl, 
  AttachedPreprocessingInfoType::iterator info);

int
instrumentRoutines(SgProject* project, bool skipTransforms);

bool
isExecutable(string expr);

void
printUsage();

ContractComment*
processCommentEntry(SgFunctionDeclaration* dNode, const string cmt);

int
processComments(SgFunctionDefinition* def);


/*
 *************************************************************************
 * Global data
 *************************************************************************
 */

static ContractComment* g_invariants;


/*
 *************************************************************************
 * Routines
 *************************************************************************
 */

/**
 * Add checks for all contract clause assertion expressions to the start
 * of the routine body.  
 *
 * @param def        Function definition.
 * @param body       Pointer to the function body.  Assumed to belong to
 *                     def.
 * @param cc         The contract clause whose expressions are to be added.
 * @param firstTime  true if this is to be the first expression
 *                   for the routine; otherwise, false.
 * @return           The number of statements added to the body.
 */
int
addPreChecks(SgFunctionDefinition* def, SgBasicBlock* body, ContractComment* cc,
  bool firstTime)
{
  int num = 0;

  if ( (def != NULL) && (body != NULL) && (cc != NULL) && (cc->size() > 0) )
  {
    ContractClauseEnum ccType = cc->clause();
    if (  (ccType == ContractClause_PRECONDITION)
       || (ccType == ContractClause_INVARIANT) )
    {
#ifdef DEBUG
      cout << "DEBUG: addPreChecks: Adding "<<cc->size()<<" expressions...\n";
#endif /* DEBUG */

      int left = cc->size();
      int last = firstTime ? 1 : 0;

      list<AssertionExpression> aeList = cc->getList();
      for(list<AssertionExpression>::iterator iter = aeList.begin();
          iter != aeList.end(); iter++)
      {
        AssertionExpression ae = (*iter);
        SgExprStatement* sttmt = buildCheck(body, ccType, (left-- == last), 
                                   ae.label(), ae.expr());
        if (sttmt != NULL)
        {
          body->prepend_statement(sttmt);
#ifdef DEBUG
          cout << "DEBUG: ...prepended: ";
          cout << sttmt->unparseToString() << "\n";
#endif /* DEBUG */
          num++;
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
 * Add checks for all contract clause assertion expressions to the end
 * of the routine body.  
 *
 * @param def        Function definition.
 * @param body       Pointer to the function body.  Assumed to belong to
 *                     def.
 * @param cc         The contract clause whose expressions are to be added.
 * @param firstTime  true if this is to be the first expression
 *                   for the routine; otherwise, false.
 * @return           The number of statements added to the body.
 */
int
addPostChecks(SgFunctionDefinition* def, SgBasicBlock* body, 
  ContractComment* cc, bool firstTime)
{
  int num = 0;

  if ( (def != NULL) && (body != NULL) && (cc != NULL) && (cc->size() > 0) )
  {
    ContractClauseEnum ccType = cc->clause();
    if (  (ccType == ContractClause_POSTCONDITION)
       || (ccType == ContractClause_INVARIANT) )
    {
#ifdef DEBUG
      cout << "DEBUG: addPostChecks: Adding "<<cc->size()<<" expressions...\n";
#endif /* DEBUG */

      int left = cc->size();
      int last = firstTime ? 1 : 0;

      list<AssertionExpression> aeList = cc->getList();
      for(list<AssertionExpression>::iterator iter = aeList.begin();
          iter != aeList.end(); iter++)
      {
        AssertionExpression ae = (*iter);
        SgExprStatement* sttmt = buildCheck(body, ccType, (left-- == last), 
                                   ae.label(), ae.expr());
        if (sttmt != NULL)
        {
          int i = SageInterface::instrumentEndOfFunction(
                   def->get_declaration(), sttmt);
          num+=i;
#ifdef DEBUG
          cout << "DEBUG: ...added to end of function: ";
          cout << sttmt->unparseToString() << "\n";
#endif /* DEBUG */
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
 * Extract and add assertion expressions to the contract clause.
 *
 * @param clause  The contract clause text extracted from the structured 
 *                  comment.
 * @param cc      The contract clause.
 */
void
addExpressions(string clause, ContractComment* cc)
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
            label = removeWS(statement.substr(0, endL));
            startE = endL+1;
#ifdef DEBUG
            cout << label + ": ";
#endif /* DEBUG */
          }
        }

        expr = removeWS(statement.substr(startE));

#ifdef DEBUG
        cout << expr << "\n";
#endif /* DEBUG */

        if (expr.find("pce_result") != string::npos )
        {
          cc->setResult(true);
#ifdef DEBUG
        cout << "DEBUG: ...contains pce_result\n";
#endif /* DEBUG */
        }

        if (isExecutable(expr)) 
        {
          AssertionExpression ae (label, expr);
          cc->add(ae);
#ifdef DEBUG
          cout << "DEBUG: ...is executable expression.\n";
#endif /* DEBUG */
        }
        else
        {
          cout << "\nWARNING: No translation support for: "<<expr<<"\n";
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
 * @param def  The function definition node.
 * @param body  Pointer to the function body.
 * @return      Returns number of finalization calls added.    
 */
int
addFinalize(SgFunctionDefinition* def, SgBasicBlock* body)
{
  int num = 0;

  if ( (def != NULL) && (body != NULL) )
  {
    /*
     * @todo TBD/FIX:  Temporarily adding statistics dump here.
     */
    SgExprListExp* parmsD = new SgExprListExp(FILE_INFO);
    if (parmsD != NULL)
    {
      parmsD->append_expression(SageBuilder::buildVarRefExp("pce_enforcer"));
      parmsD->append_expression(new SgStringVal(FILE_INFO, "End processing"));
      SgExprStatement* sttmt = SageBuilder::buildFunctionCallStmt(
        "ContractsEnforcer_dumpStatistics", SageBuilder::buildVoidType(), 
        parmsD, body);
      if (sttmt != NULL)
      {
        attachTranslationComment(sttmt, "Contract Enforcement Data Dump");
        int i = SageInterface::instrumentEndOfFunction(def->get_declaration(),
                   sttmt);
        num+=i;
      }
    }

    SgExprListExp* parmsF = new SgExprListExp(FILE_INFO);
    if (parmsF != NULL)
    {
      SgExprStatement* sttmt = SageBuilder::buildFunctionCallStmt(
        "ContractsEnforcer_finalize", SageBuilder::buildVoidType(), 
        parmsF, body);
      if (sttmt != NULL)
      {
        attachTranslationComment(sttmt, "Contract Enforcement Finalization");
        int i = SageInterface::instrumentEndOfFunction(def->get_declaration(),
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
 * @param project         The Sage project representing the initial AST of the
 *                          file(s).
 * @param skipTransforms  True if the transformations are to be skipped;
 *                          otherwise, false.
 * @return                The processing status: 0 for success, non-0 for 
 *                          failure.
 */
int
addIncludes(SgProject* project, bool skipTransforms)
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
          string contractsHdr = "#include \"contracts.h\"";
          string enforcerHdr  = "#include \"ContractsEnforcer.h\"";
          string optionsHdr   = "#include \"contractOptions.h\"";
          string includes     = contractsHdr + "\n" + enforcerHdr + "\n"
            + optionsHdr;

          SageInterface::attachComment(globalScope,
            "Added contract enforcement includes.", PreprocessingInfo::before,
            PreprocessingInfo::C_StyleComment);
    
          SageInterface::addTextForUnparser(globalScope, includes,
            AstUnparseAttribute::e_after);
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
 * @return       Number of initialization calls added.
 */
int
addInitialize(SgBasicBlock* body)
{
  int num = 0;

  if (body != NULL)
  {
    SgExprListExp* parms = new SgExprListExp(FILE_INFO);
    if (parms != NULL)
    {
      parms->append_expression(SageBuilder::buildVarRefExp("NULL"));
      SgExprStatement* sttmt = SageBuilder::buildFunctionCallStmt(
        "ContractsEnforcer_initialize", SageBuilder::buildVoidType(), 
        parms, body);
      if (sttmt != NULL)
      {
        attachTranslationComment(sttmt, "Contract Enforcement Initialization");
        body->prepend_statement(sttmt);
        num += 1;
      }
    }
  }
              
  return num;
}  /* addInitialize */


/**
 * Build the contract clause check statement.
 *
 * @param body       Pointer to the function body.
 * @param cc         The contract clause whose expressions are to be added.
 * @param firstTime  true if this is to be the first expression
 *                   for the routine; otherwise, false.
 * @return           Contract clause statement node.
 */
SgExprStatement*
buildCheck(SgBasicBlock* body, ContractClauseEnum clauseType, bool firstTime,
           string label, string expr)
{
  SgExprStatement* sttmt = NULL;

  if ( (body != NULL) && !expr.empty() )
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
            cmt = "PContract Precondition";
          }
          break;
        case ContractClause_POSTCONDITION:
          {
            parms->append_expression(SageBuilder::buildVarRefExp(
              "ContractClause_POSTCONDITION"));
            clauseTime = "pce_def_times.post";
            cmt = "PContract Postcondition";
          }
          break;
        case ContractClause_INVARIANT:
          {
            parms->append_expression(SageBuilder::buildVarRefExp(
              "ContractClause_INVARIANT"));
            clauseTime = "pce_def_times.inv";
            cmt = "PContract Invariant";
          }
          break;
        default:
          {
            /*  WARNING:  This should NEVER happen... */
            parms->append_expression(SageBuilder::buildVarRefExp(
              "ContractClause_NONE"));
            clauseTime = "0";
            cmt = "PContract Error:  Contact Developer(s)!";
          }
          break;
      }

      parms->append_expression(SageBuilder::buildVarRefExp(clauseTime));
      parms->append_expression(SageBuilder::buildVarRefExp(
        "pce_def_times.routine"));
      parms->append_expression(SageBuilder::buildVarRefExp(
        (firstTime) ? "CONTRACTS_TRUE" : "CONTRACTS_FALSE"));
      parms->append_expression(new SgStringVal(FILE_INFO, label));
      parms->append_expression(SageBuilder::buildVarRefExp("(" + expr + ")"));
  
      sttmt = SageBuilder::buildFunctionCallStmt("PCE_CHECK_EXPR_TERM", 
        SageBuilder::buildVoidType(), parms, body);
      if (sttmt != NULL)
      {
        attachTranslationComment(sttmt, cmt);
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
      cout << "DEBUG: buildCheck: New parameters is NULL.\n";
    }
#endif /* DEBUG */
  }
#ifdef DEBUG
  else
  {
    if (body == NULL) cout << "DEBUG: buildCheck: NULL body\n";
    if (expr.empty()) cout << "DEBUG: buildCheck: Empty expression\n";
  }
#endif /* DEBUG */
              
  return sttmt;
}  /* buildCheck */


/**
 * Extract the contract clause, if any, from the pre-processing directive.
 *
 * @param dType  The pre-processing directive type.
 * @param cc     The resulting contract clause, if any.
 * @return       The ContractComment type.
 */
ContractComment*
extractContractClause(SgFunctionDeclaration* decl, 
                      AttachedPreprocessingInfoType::iterator info)
{
  ContractComment* cc = NULL;

  if ((*info) != NULL)
  {
    switch ((*info)->getTypeOfDirective())
    {
      case PreprocessingInfo::C_StyleComment:
        {
          string str = (*info)->getString();
          cc = processCommentEntry(decl, str.substr(2, str.size()-4));
        }
        break;
      case PreprocessingInfo::CplusplusStyleComment:
        {
          string str = (*info)->getString();
          cc = processCommentEntry(decl, str.substr(2));
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
    cout << "DEBUG:  extractContractClause: Information is NULL\n";
  }
#endif /* DEBUG */

  return cc;
}  /* extractContractClause */


/**
 * Add (test) contract assertion checks to each routine.
 *
 * @param project         The Sage project representing the initial AST of the
 *                          file(s).
 * @param skipTransforms  True if the transformations are to be skipped;
 *                          otherwise, false.
 * @return                The processing status: 0 for success, non-0 for 
 *                          failure.
 */
int
instrumentRoutines(SgProject* project, bool skipTransforms)
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
        SgFunctionDeclaration* decl;
        SgFunctionDefinition* def = isSgFunctionDefinition(*iter);

        /*
         * Get all of the attendant information and ensure the node
         * is ACTUALLY to be associated with generated output.
         */
        if (  (def != NULL) 
           && ((info=def->get_file_info()) != NULL)
           && isInputFile(project, info->get_raw_filename())
           && ((decl=def->get_declaration()) != NULL) )
        {
          SgFunctionDeclaration* defDecl =
            isSgFunctionDeclaration(decl->get_definingDeclaration());

          if ( (defDecl != NULL) && (defDecl == decl) )
          { 
            if (skipTransforms)
            {
              cout<<"\n"<<++num<<": "<<info->get_raw_filename()<<":\n   ";
              cout<<getBasicSignature(defDecl)<<endl;
            }
            else
            {
              num += processComments(def);
            }
          }
        }
      }

      cout<<"Added "<<num<<" contract checks, initialization, and/or ";
      cout<<"finalization statements.\n";
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
 * @param expr  The string representing the assertion expression.
 * @return      True if the expression appears to be executable; False 
 *                otherwise.
 */
bool
isExecutable(string expr)
{
  bool isOkay = true;

  if (expr == "is pure") 
  {
    isOkay = false;
#ifdef DEBUG
    cout << "DEBUG: Skipping 'is pure'...\n";
#endif /* DEBUG */
  }
  else
  {
    for (int i=MIN_NEE_INDEX; i<MAX_NEE_INDEX; i++)
    {
      if (expr.find(NonExecExpressions[i]) != string::npos)
      {
#ifdef DEBUG
        cout << "DEBUG: Determined "<<NonExecExpressions[i]<<" in "<<expr;
        cout << ": Non-executable\n";
#endif /* DEBUG */
        isOkay = false;
        break;
      }
    }
  }

  return isOkay;
}  /* isExecutable */


/**
 * Print usage information (i.e., how to run the executable).
 */
void
printUsage()
{
  cout << "\nUSAGE:  ContractInstrumenter [option] <source-file-list>\n\n";
  cout << "where option can include one or more basic ROSE options, such as:\n";
  cout << "  -rose:verbose [LEVEL]\n";
  cout << "              Verbosely list internal processing, with higher\n";
  cout << "              levels generating more output (default 0).\n";
  cout << "  -rose:skip_transformation\n";
  cout << "              Read input file but skip all transformations.\n";
  cout << "and\n";
  cout << "  <source-file-list>  is a list of one or more source file names.\n";
  return;
}  /* printUsage */


/**
 * Process the comment to assess and handle any contract annotation.
 *
 * @param dNode  Current AST node.
 * @param cmt    Comment contents.
 * @return       The corresponding ContractComment type.
 */
ContractComment*
processCommentEntry(SgFunctionDeclaration* dNode, const string cmt)
{
  ContractComment* cc = NULL;

  if ( (dNode != NULL) && !cmt.empty() )
  {
    size_t pos;
    if ((pos=cmt.find("CONTRACT"))!=string::npos)
    {
      if ((pos=cmt.find("REQUIRE"))!=string::npos)
      {
        cc = new ContractComment(ContractComment_PRECONDITION);
        addExpressions(cmt.substr(pos+7), cc);
      }
      else if ((pos=cmt.find("ENSURE"))!=string::npos)
      {
        cc = new ContractComment(ContractComment_POSTCONDITION);
        addExpressions(cmt.substr(pos+6), cc);

      }
      else if ((pos=cmt.find("INVARIANT"))!=string::npos)
      {
        cc = new ContractComment(ContractComment_INVARIANT);
        addExpressions(cmt.substr(pos+9), cc);
      }
      else if ((pos=cmt.find("INIT"))!=string::npos)
      {
        cc = new ContractComment(ContractComment_INIT);
      }
      else if ((pos=cmt.find("FINAL"))!=string::npos)
      {
        cc = new ContractComment(ContractComment_FINAL);
      }
      else
      {
        string msg = "WARNING: Unidentified contract annotation: ";
        printLineComment(dNode, msg + cmt.substr(pos+8));
      }
    }
  }
  return cc;
} /* processCommentEntry */


/**
 * Process comments associated with the function definition.
 *
 * @param def  The function definition node.
 * @return     The number of statements added.
 */
int
processComments(
SgFunctionDefinition* def)
{
  int num = 0;

  if (def != NULL)
  {
    SgFunctionDeclaration* decl = def->get_declaration();
    if (decl != NULL)
    {
      bool isConstructor = false;
      bool isDestructor = false;
      SgMemberFunctionDeclaration* mfDecl = isSgMemberFunctionDeclaration(decl);
      if (mfDecl != NULL)
      {
        SgSpecialFunctionModifier sfMod = mfDecl->get_specialFunctionModifier();
        isConstructor = sfMod.isConstructor();
        isDestructor = sfMod.isDestructor();
      }

      SgName nm = decl->get_name();

      AttachedPreprocessingInfoType* cmts = 
        decl->getAttachedPreprocessingInfo();
      if (cmts != NULL)
      {
        ContractComment* pre = NULL;
        ContractComment* post = NULL;
        bool hasInit = false;
        bool hasFinal = false;
        int numInvariants = (g_invariants != NULL) ? g_invariants->size() : 0;
        int numChecks[] = { 0, 0, numInvariants };
        int numPrep[] = { 0, 0 };

        AttachedPreprocessingInfoType::iterator iter;
        for (iter = cmts->begin(); iter != cmts->end(); iter++)
        {
          ContractComment* cc = extractContractClause(decl, iter);

          if (cc != NULL)
          {
            switch (cc->type())
            {
            case ContractComment_PRECONDITION:
              {
                pre = cc;
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
                if (g_invariants == NULL)
                {
                  g_invariants = cc;
                  numChecks[2] += g_invariants->size();
                }
                else
                {
                  delete cc;
                  cout<<"\nWARNING: Ignoring additional invariant clause: "<<nm;
                  cout<<"\n";
                }
              }
              break;
            case ContractComment_INIT:
              {
                hasInit = true;
                delete cc;
              }
              break;
            case ContractComment_FINAL:
              {
                hasFinal = true;
                delete cc;
              }
              break;
            case ContractComment_NONE:
            default:
              {
                delete cc;
              }
              break;
            } /* end switch */
          } /* end if have contract comment to process */
        } /* end for each comment */

        if (  hasInit || hasFinal || (pre != NULL) || (post != NULL)
           || (g_invariants != NULL) )
        {
          SgBasicBlock* body = def->get_body();
  
          if (body != NULL)
          {
            /*
             * First add initial routine instrumentation.
             * ...Order IS important since each is prepended to the body.
             */
            bool isFirst = numChecks[2] <= 0;
            if (pre != NULL) 
            { 
              if (numChecks[0] > 0) 
              {
                num += addPreChecks(def, body, pre, isFirst);
                isFirst = false;
              }
              delete pre;
            }

            if ( (g_invariants != NULL) && (numChecks[2] > 0) )
            {
              if (!isConstructor)
              {
              ///// @todo: TODO/FIX:  Why are invariants being skipped
              /////    on subsequent routines?
                num += addPreChecks(def, body, g_invariants, true);
                isFirst = false;
              }
            }

            if (hasInit)
            {
              numPrep[0] += addInitialize(body);
              num += numPrep[0];
            }

            /*
             * Now add post-routine checks.
             */
            if (post != NULL) 
            {
              if (numChecks[1] > 0)
              {
                num += addPostChecks(def, body, post, isFirst);
                isFirst = false;
              }
              delete post;
            }

            if ( (g_invariants != NULL) && (numChecks[2] > 0) )
            {
              if (!isDestructor) {
                num += addPostChecks(def, body, g_invariants, false);
              } 
            } 

            if (hasFinal)
            {
              numPrep[1] += addFinalize(def, body);
              num += numPrep[1];
            }
          } /* end if have an annotation destination */
        } /* end if have annotations to make */

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
        cout<<"DEBUG:END ************************************\n";
#endif /* DEBUG */
      } /* end if comments */
    } /* end if have declaration */
  } /* end if have definition */

  return num;
}  /* processComments */


/**
 * Build and process AST nodes of input source files.
 */
int
main(int argc, char* argv[])
{
  int status = 0;

  if (argc > 1)
  {
    /* Build initial (ROSE) AST. */
    SgProject* project = frontend(argc, argv);

    if (project != NULL)
    {
      /* Prepare to honor the ROSE transformation command line option. */
      bool skipTransforms = project->get_skip_transformation();
  
      if (skipTransforms)
        cout << "WARNING:  Skipping transformations per ROSE option.\n\n";
  
      /* First add requisite include files. */
      status = addIncludes(project, skipTransforms);
  
      /* Now instrument routines. */
      if (status == 0)
      {
        status = instrumentRoutines(project, skipTransforms);
        if (g_invariants != NULL)
        {
          delete g_invariants;
        }
      }
      else
      {
        cerr << "ERROR: Skipping routines instrumentation call due to ";
        cerr << "previous error(s).\n";
      }

      delete project;
    }
    else
    {
      cerr << "\nERROR:  Failed to build the AST.\n";
      status = 1;
    }
  }
  else 
  {
    printUsage();
    status = 1;
  }

  return status;
}  /* main */
