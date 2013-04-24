/**
 * \internal
 * File:           RoutineContractInstrumenter.cpp
 * Author:         T. Dahlgren
 * Created:        2012 August 3
 * Last Modified:  2013 April 23
 * \endinternal
 *
 * @file
 * @brief
 * Initial contract enforcement instrumentation example.
 *
 * @details
 * This example only translates structured contract comments associated
 * with function nodes in the AST into either contract checks or comments,
 * depending on whether a rudimentary (string) check of the associated 
 * expression indicates it is (likely to be) a basic C expression or a more
 * advanced expression using special operators, keywords, or functions.
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
        cout << "WARNING: Skipping transformations per ROSE option.\n\n";
  
      /* First add requisite include files. */
      ContractsProcessor processor = ContractsProcessor();
      status = processor.addIncludes(project, skipTransforms);
  
      /* Now instrument routines. */
      if (status == 0)
      {
        status = processor.instrumentRoutines(project, skipTransforms);
        /*
         * The following appears to break on C++ sources:
         *
        if (status == 0)
        {
          status = backend(project);
        } 
        */
      }
      else
      {
        cerr << "\nERROR: Skipping routines instrumentation call due to ";
        cerr << "previous error(s).\n";
      }

      delete project;
    }
    else
    {
      cerr << "\nERROR: Failed to build the AST.\n";
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
