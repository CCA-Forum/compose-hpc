/**
 * File:          ContractPrinter.cpp
 * Author:        T. Dahlgren
 * Created:       2012 July 6
 * Last Modified: 2012 November 12
 *
 *
 * @file
 * @section DESCRIPTION
 * Simple class implementation for illustrating the use of basic ROSE features 
 * for printing contract clause comments.  It is NOT intended for contract
 * enforcement use.
 *
 * @todo Clean up this example so this class can be re-used "properly"
 * in ContractAssertionPrinter.cpp.
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
#include "ContractPrinter.hpp"
#include "RoseHelpers.hpp"


using namespace std;


void
ContractPrinter::visit(SgNode* node)
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
        size_t cInd;
        if (isCComment(((*iter)->getTypeOfDirective())))
        {
          string str = (*iter)->getString();
          if (str.find("CONTRACT")!=string::npos)
          {
            if (str.find("REQUIRE")!=string::npos)
            {
              printLineComment(node, "Precondition clause:");
              cout<<(*iter)->getString()<<endl;
            }
            else if (str.find("ENSURE")!=string::npos)
            {
              printLineComment(node, "Postcondition clause:");
              cout<<(*iter)->getString()<<endl;
            }
            else if (str.find("INVARIANT")!=string::npos)
            {
              printLineComment(node, "Invariant clause:");
              cout<<(*iter)->getString()<<endl;
            }
            else if (str.find("INIT")!=string::npos)
            {
              printLineComment(node, "Initialization:");
              cout<<(*iter)->getString()<<endl;
            }
            else if (str.find("FINAL")!=string::npos)
            {
              printLineComment(node, "Finalization:");
              cout<<(*iter)->getString()<<endl;
            }
            else
            {
              printLineComment(node, 
                "WARNING: Unidentified contract annotation:");
              cout<<(*iter)->getString()<<endl;
            }
          }
        }
      }
      cout<<endl;
    }

    /* Do NOT attempt to delete lNode or cmts! */
  }
  return;
} /* ContractPrinter::visit */


#ifndef NO_MAIN
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
    /* Build the traversal object. */
    ContractPrinter* vis = new ContractPrinter();

    /*
     * Traverse each input file, starting at project node of the AST
     * and using preorder traversal.
     */
    vis->traverseInputFiles(project, preorder);

    delete vis;
    delete project;
  } else {
    cout << "\nERROR:  Failed to build the AST.\n";
    status = 1;
  }

  return status;
} /* main */
#endif /* NO_MAIN */

