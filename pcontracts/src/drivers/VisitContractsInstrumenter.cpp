/**
 * \internal
 * File:          VisitContractsInstrumenter.cpp
 * Author:        T. Dahlgren
 * Created:       2012 November 9
 * Last Modified: 2013 August 2
 * \endinternal
 *
 * @file
 * @brief
 * Second contract enforcement instrumentation example.
 *
 * @details
 * This is the main driver for translating structured contract comments 
 * associated with any AST node into contract checks, contract-related 
 * runtime calls, or comments, depending on the annotation and associated 
 * AST node.  Preconditions and Postconditions must still be associated 
 * with function nodes; however, all other contract annotations can be 
 * associated with other node types.
 *
 * Only rudimentary (string) check of the associated expression is performed to
 * assess whether the expression is (likely to be) a basic C expression or a
 * more advanced expression containing special operators, keywords, or 
 * functions.
 *
 * @bug  The helloworld-v2.cc example illustrates a case where ROSE (17286) 
 *  fails to provide an AST node for the final 'return 0;', which has the
 *  inlined 'CONTRACT FINAL' annotation.  As a result, the visitor is
 *  unable to add the contract finalization call to the routine.
 *  The current "work around" is to either NOT inline the annotation
 *  or ensure there's some extra functionality performed between the
 *  "finalization" annotation and the return.  helloworld-v3.cc was
 *  added to illustrate this point.
 * 
 *
 * @todo Generate return variable when needed.
 *
 * @todo Support configuration of enforcer and execution time estimates
 *  for contract clauses and routines.  Ensure time estimates are properly
 *  utilized in relevant partial enforcement options.
 *
 * @todo Support of advanced (ie, non-C) expressions requires parsing 
 *  expressions.  This includes the equivalent of SIDL built-in assertion
 *  expression "routines".
 *
 * @todo Consider ensuring class invariants only appear prior to function
 *  definitions.
 *
 * @todo Give some thought to improving contract violation messages.
 *
 * @todo Add a new annotation for checking/dumping contract check data;
 *  ensure configuration is being done properly; and test the feature.
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
#include "VisitContractsInstrumenter.hpp"
#include "ContractsProcessor.hpp"

using namespace std;


VisitContractsInstrumenter::VisitContractsInstrumenter(Sg_File_Info* fileInfo) 
  : d_processor(new ContractsProcessor()), d_fileInfo(fileInfo), d_lastLine(-1),
    d_num(0) {}


VisitContractsInstrumenter::~VisitContractsInstrumenter()
{
  delete d_processor;
}

void
VisitContractsInstrumenter::visit(
  /* inout */ SgNode* node)
{
  SgLocatedNode* lNode = isSgLocatedNode(node);
  if (lNode != NULL)
  {
    Sg_File_Info* info = lNode->get_file_info();
    long line;
    if (  (info != NULL) && (info->isSameFile(d_fileInfo)) 
       && ( (line = info->get_raw_line()) != d_lastLine )
       )
    {
      d_lastLine = line;

#ifdef DEBUG
      printLineComment(lNode, "\nDEBUG:  Processing...", true);
      cout << "Node type: " << lNode->variantT() << "(";
      cout << Cxx_GrammarTerminalNames[lNode->variantT()].name << ")\n";
#endif /* DEBUG */

      SgGlobal* globalScope = NULL;
      SgFunctionDefinition* def = NULL;
      SgFunctionDeclaration* decl = NULL;

      if ( (globalScope = isSgGlobal(node)) != NULL )
      {
        int status = d_processor->addIncludes(globalScope);
      }
      else if ( (decl = isSgFunctionDeclaration(lNode)) != NULL )
      {
        def = decl->get_definition();
        if (def != NULL)
        {
          d_num += d_processor->processFunctionDef(def);
        }
#ifdef PCE_ENABLE_WARNINGS
        else
        {  
          cout << "\nWARNING: Detected null definition for a declaration.\n  ";
          cout << "Ignoring any associated contract comments.\n";
        }
#endif /* PCE_ENABLE_WARNINGS */
      }
      else if ( (def = isSgFunctionDefinition(lNode)) == NULL )
      {
        /* The node could support an invariant or contract clause. */
        d_num += d_processor->processNonFunctionNode(lNode);
      }
#ifdef DEBUG
      else
      {
         cout << "Skipping Node type: " << lNode->variantT() << "(";
         cout << Cxx_GrammarTerminalNames[lNode->variantT()].name << ")\n";
      }
#endif /* DEBUG */
    }
  }
  return;
} /* visit */


/**
 * Output the number of contract-related statements added, if any.
 */
void
VisitContractsInstrumenter::atTraversalEnd(void)
{
  cout<<"\nAdded "<<d_num<<" contract-related statements.\n";
  return;
}  /* atTraversalEnd */


/**
 * Build and process the AST nodes of the input source file(s).
 */
int 
main(int argc, char* argv[])
{
  int status = 0;

  /* Building the initial AST. */
  SgProject* project = frontend(argc, argv);

  if (project != NULL)
  {
    /* Check internal consistency of the AST.  */
    AstTests::runAllTests(project);

    /*
     * Ensure we have a SINGLE file.
     */
    if (project->numberOfFiles() == 1)
    {
      SgFile* file = project->get_fileList()[0];
      if (file != NULL)
      {
        /* Warn the user the ROSE skip transformation option will be ignored. */
        if (project->get_skip_transformation())
        {
          cout<<"WARNING: The skip transformation option is NOT honored.\n\n";
        }

        Sg_File_Info* info = file->get_file_info();
        if (info != NULL)
        {
          VisitContractsInstrumenter* vis = 
              new VisitContractsInstrumenter(info);

          if (vis != NULL)
          {
            vis->traverseInputFiles(project, preorder);


            /*
             * The following is REQUIRED to generate source (unlike
             * the routine instrumentation version).
             */
            status = backend(project);

            delete vis;
          }
          /* Do NOT attempt to delete info as doing so messes up AST. */
        }
      } else {
        cout << "\nERROR: Failed to retrieve the file node.\n";
        status = 1;
      }
    } else {
      cout << "\nERROR: Only ONE file can be processed at a time.\n";
      status = 1;
    }

    delete project;
  } else {
    cout << "\nERROR: Failed to build the AST.\n";
    status = 1;
  }

  return status;
} /* main */
