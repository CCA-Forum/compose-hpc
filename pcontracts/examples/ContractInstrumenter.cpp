/**
 * File:           ContractInstrumenter.cpp
 * Author:         T. Dahlgren
 * Created:        2012 August 3
 * Last Modified:  2012 August 17
 *
 * @file
 * @section DESCRIPTION
 * Experimental contract enforcement instrumentation.
 *
 *
 * @todo Translate instrumentation arguments to the proper node types.
 *
 * @todo Change this to a visitor since must extract actual contract
 *  clauses.
 *
 * @todo Use SgGlobal to insert required includes.
 *
 * @todo Write and instrument routines for instantiating the enforcer 
 *  "instance".
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
#include <string>
#include "rose.h"
#include "Cxx_Grammar.h"
#include "RoseHelpers.hpp"
#include "contractOptions.h"

#define FILE_INFO Sg_File_Info::generateDefaultFileInfoForTransformationNode()

using namespace std;


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
 * Get the basic signature.  
 *
 * @param decl  The function declaration
 * @return      A basic signature derived from the node's unparsed output.
 */
string
getBasicSignature(string decl)
{
  string res;

  size_t bst = decl.find_first_of("{");
  if (bst!=string::npos)
  {
    res.append(decl.substr(0, bst));
    res.append(";");
  }
  else
  {
    cerr<<"\nERROR:  Failed to locate starting (body) brace: "<<decl<<endl;
  }
  
  return res;
}  /* getBasicSignature */


/**
 * Build contract clause check statement.
 *
 * @param body       Pointer to the function body.
 * @param clauseType Type of the instrumented contract clause.
 * @param firstTime  true if this is to be the first expression
 *                   for the routine; otherwise, false.
 * @param label      The assertion expression label to be included
 *                     in the violation error message.
 * @param expr       The assertion expression.
 * @return           Contract clause statement node.
 */
SgExprStatement*
buildCheck(SgBasicBlock* body, ContractClauseEnum clauseType, bool firstTime,
           string label, string expr)
{
  SgExprStatement* sttmt = NULL;

  if ( (body != NULL) && !label.empty() && !expr.empty() )
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
      parms->append_expression(SageBuilder::buildVarRefExp(expr));
  
      sttmt = SageBuilder::buildFunctionCallStmt("PCE_CHECK_EXPR_TERM", 
        SageBuilder::buildVoidType(), parms, body);
      attachTranslationComment(sttmt, cmt);
    }
  }
              
  return sttmt;
}  /* buildCheck */


/**
 * Add (test) contract assertion checks to each routine.
 *
 * @param project   The Sage project representing the initial AST of the
 *                    file(s).
 * @return          The processing status: 0 for success, non-0 for failure.
 */
int
instrumentRoutines(SgProject* project)
{
  int status = 0;

  if (project != NULL)
  {
    /* Prepare to honor the ROSE transformation command line option. */
    bool skipTransforms = project->get_skip_transformation();

    /* Find all function definitions. */
    vector<SgNode*> fdList = 
      NodeQuery::querySubTree(project, V_SgFunctionDefinition);

    if (!fdList.empty())
    {
      int num = 0;

      if (skipTransforms)
        cout << "WARNING:  Skipping transformations per ROSE option.\n\n";

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
              cout<<getBasicSignature(defDecl->unparseToString())<<endl;
            }
            else
            {
              SgName nm = defDecl->get_name();
              SgBasicBlock* body = def->get_body();
  
              /* Build and insert precondition and/or invariant check(s). */
              SgExprStatement* sttmt = buildCheck(body, 
                ContractClause_PRECONDITION, false, 
                "testPre", "CONTRACTS_TRUE" );
              if (sttmt != NULL)
              {
                body->prepend_statement(sttmt);
                num++;
              }
              sttmt = buildCheck(body, ContractClause_INVARIANT, true, 
                "testInv", "CONTRACTS_TRUE" );
              if (sttmt != NULL)
              {
                body->prepend_statement(sttmt);
                num++;
              }
  
              /* Build and insert postcondition and/or invariant check(s). */
              sttmt = buildCheck(body, ContractClause_INVARIANT, false, 
                "testInv", "CONTRACTS_TRUE" );
              if (sttmt != NULL)
              {
                int i = SageInterface::instrumentEndOfFunction(
                         def->get_declaration(), sttmt);
                num+=i;
              }
              sttmt = buildCheck(body, ContractClause_POSTCONDITION, false, 
                "testPost", "CONTRACTS_TRUE" );
              if (sttmt != NULL)
              {
                int i = SageInterface::instrumentEndOfFunction(
                         def->get_declaration(), sttmt);
                num+=i;
              }

#ifdef DEBUG
              cout<<nm.getString()<<"():  Instrumented routine: begin ";
              cout<<"and "<<i<<" return(s).\n";
#endif /* DEBUG */
            }
          }
        }
      }

      cout<<"Added "<<num<<" (test) assertion expression checks.\n";
    }

    /* Run consistency checks (?) */
    //AstTests::runAllTests(project);

    /* Translate the file(s) */
    project->unparse();

#ifdef DEBUG
    cout<<"Processed "<<fdList.size()<<" function definitions.\n\n";
#endif /* DEBUG */
    //delete fdList;
    delete project;
  }
  else
  {
    cout << "\nERROR:  Failed to build the AST.\n";
    status = 1;
  }

  return status;
}  /* instrumentRoutines */


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

    /* Now instrument routines. */
    status = instrumentRoutines(project);
  }
  else 
  {
    printUsage();
    status = 1;
  }

  return status;
}  /* main */
