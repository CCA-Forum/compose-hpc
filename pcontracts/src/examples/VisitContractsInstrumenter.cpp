/**
 * \internal
 * File:          VisitContractsInstrumenter.cpp
 * Author:        T. Dahlgren
 * Created:       2012 November 9
 * Last Modified: 2013 January 22
 * \endinternal
 *
 * @file
 * @brief
 * Second contract enforcement instrumentation example.
 *
 * @details
 * This example translates structured contract comments associated with any
 * AST node into contract checks, contract-related runtime calls, or comments,
 * depending on the annotation and associated AST node.  Preconditions
 * and Postconditions must still be associated with function nodes; however, 
 * all other contract annotations can be associated with other node types.
 *
 * Only rudimentary (string) check of the associated expression is performed to
 * assess whether the expression is (likely to be) a basic C expression or a
 * more advanced expression containing special operators, keywords, or 
 * functions.
 *
 *
 * @todo Finish support for non-function definition nodes.
 * 
 * @todo Determine if "firstTime" handling is working now.  If not, fix it.
 *
 * @todo Generate return variable when needed.
 *
 * @todo Support configuration of enforcer and execution time estimates
 *  for contract clauses and routines.  Ensure time estimates are properly
 *  utilized in relevant partial enforcement options.
 *
 * @todo Ensure proper handling of clause checks and consider limiting
 *  BEGIN and END comments surrounding clauses in cases where multiple
 *  expressions are present.
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
 * @todo Consider checking language and attaching comment of corresponding
 *  style.
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


void
VisitContractsInstrumenter::visit(
  /* inout */ SgNode* node)
{
  SgLocatedNode* lNode = isSgLocatedNode(node);
  if (lNode != NULL)
  {
    SgGlobal* globalScope;
    SgFunctionDefinition* def;
    if ( (globalScope = isSgGlobal(node)) != NULL )
    {
      addIncludeFiles(globalScope);
    }
    else if ( (def = isSgFunctionDefinition(lNode)) != NULL )
    {
      d_num += d_processor.processFunctionDef(def);
    }
    else /* The node could support an invariant or contract clause. */
    {
      d_num += d_processor.processGeneralNode(lNode);
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
    /* Warn the user the ROSE skip transformation option will be ignored. */
    bool skipTransforms = project->get_skip_transformation();

    if (skipTransforms)
      cout << "WARNING:  The skip transformation option is NOT honored.\n\n";

    /* Build the traversal object. */
    VisitContractsInstrumenter* vis = new VisitContractsInstrumenter();

    if (vis != NULL)
    {
      /*
       * Traverse each input file, starting at project node of the AST
       * and using preorder traversal.
       */
      vis->traverseInputFiles(project, preorder);

      delete vis;
    }

    delete project;
  } else {
    cout << "\nERROR:  Failed to build the AST.\n";
    status = 1;
  }

  return status;
} /* main */
