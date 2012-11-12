/**
 * File:           RoutineContractInstrumenter.cpp
 * Author:         T. Dahlgren
 * Created:        2012 August 3
 * Last Modified:  2012 November 12
 *
 *
 * @file
 * @section DESCRIPTION
 * Initial contract enforcement instrumentation example.
 *
 * This example only translates structured contract comments associated
 * with function nodes in the AST into either contract checks or comments,
 * depending on whether a rudimentary (string) check of the associated 
 * expression indicates it is (likely to be) a basic C expression or more
 * advanced expressions using special operators, keywords, or functions.
 *
 *
 * @section WARNING
 * This is a VERY preliminary draft. (See todo items.)
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
#include <list>
#include <string>
#include "rose.h"
#include "Cxx_Grammar.h"
#include "RoseHelpers.hpp"
#include "contractOptions.h"
#include "contractClauseTypes.hpp"
#include "ContractsProcessor.hpp"

using namespace std;


/**
 * Print usage information (i.e., how to run the executable).
 */
void
printUsage()
{
  cout << "\nUSAGE:\n";
  cout << "  RoutineContractInstrumenter [option] <source-file-list>\n\n";
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
      ContractsProcessor processor = ContractsProcessor();
      status = processor.addIncludes(project, skipTransforms);
  
      /* Now instrument routines. */
      if (status == 0)
      {
        status = processor.instrumentRoutines(project, skipTransforms);
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
