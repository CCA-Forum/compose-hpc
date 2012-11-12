/**
 * File:          VisitContractsInstrumenter.cpp
 * Author:        T. Dahlgren
 * Created:       2012 November 9
 * Last Modified: 2012 November 12
 *
 *
 * @file
 * @section DESCRIPTION
 * Experimental PAUL contracts instrumenter (via ROSE visitor pattern).
 *
 * This example translates structured contract comments according to the type
 * of AST node.  Preconditions and Postconditions must still be associated
 * with function nodes; however, all of the other contract annotations can
 * be associated with other node types.
 *
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
 *
 * @section COPYRIGHT
 * Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Tamara Dahlgren <dahlgren1@llnl.gov>.
 * 
 * LLNL-CODE-473891.
 * All rights reserved.
 * 
 * This software is part of COMPOSE-HPC. See http://compose-hpc.sourceforge.net/
 * for details.  Please read the COPYRIGHT file for Our Notice and for the 
 * BSD License.
 */

#include <iostream>
#include <string>
#include "rose.h"
#include "Cxx_Grammar.h"
#include "RoseHelpers.hpp"
#include "VisitContractsInstrumenter.hpp"

using namespace std;


int
VisitContractsInstrumenter::addIncludes(SgProject* project)
{
  int status = d_processor.addIncludes(project, false);

  if (status != 0)
    cerr << "ERROR: Failed to add include files.\n";

  return status;
} /* addIncludes */


void
VisitContractsInstrumenter::visit(SgNode* node)
{
  SgLocatedNode* lNode = isSgLocatedNode(node);
  if (lNode != NULL)
  {
    AttachedPreprocessingInfoType* cmts = lNode->getAttachedPreprocessingInfo();
    if (cmts != NULL)
    {
      AttachedPreprocessingInfoType::iterator iter;
      for (iter = cmts->begin(); iter != cmts->end(); iter++)
      {
        switch ((*iter)->getTypeOfDirective())
        {
          case PreprocessingInfo::C_StyleComment:
            {
              string str = (*iter)->getString();
              processCommentContents(node, str.substr(2, str.size()-4));
            }
            break;
          case PreprocessingInfo::CplusplusStyleComment:
            {
              string str = (*iter)->getString();
              processCommentContents(node, str.substr(2));
            }
            break;
/*
 * Fortran contract comments aren't really supported.
 *
          case PreprocessingInfo::FortranStyleComment:
          case PreprocessingInfo::F90StyleComment:
            {
              string str = (*iter)->getString();
              processCommentContents(node, str.substr(1));
            }
            break;
*/
        }
      }
    }

    /* Do NOT attempt to delete lNode or cmts! */
  }
  return;
} /* visit */

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
      cout << "WARNING:  The skip transformation option is NOT honored.\n\n"

    /* Build the traversal object. */
    VisitContractsInstrumenter* vis = new VisitContractsInstrumenter();

    if (vis != NULL)
    {
      /* Add requisite include file(s). */
      status = vis->addIncludes(project);

      if (status != 0)
      {
        /*
         * Traverse each input file, starting at project node of the AST
         * and using preorder traversal.
         */
        vis->traverseInputFiles(project, preorder);
      }

      delete vis;
    }

    delete project;
  } else {
    cout << "\nERROR:  Failed to build the AST.\n";
    status = 1;
  }

  return status;
} /* main */

