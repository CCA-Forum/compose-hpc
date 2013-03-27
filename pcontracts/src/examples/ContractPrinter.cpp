/**
 * \internal
 * File:          ContractPrinter.cpp
 * Author:        T. Dahlgren
 * Created:       2012 July 6
 * Last Modified: 2013 March 19
 * \endinternal
 *
 * @file
 * @brief 
 * Class for printing contract clause comments via ROSE AST.
 *
 * @details
 * Simple class implementation for illustrating the use of basic ROSE features 
 * for identifying and printing contract clause comments.  
 *
 * @todo (Low Priority) Clean up this example so this class can be re-used 
 * "properly" in ContractAssertionPrinter.cpp (i.e., without resorting to the 
 * #define around main()).
 *
 * @htmlinclude copyright.html
 */

#include <iostream>
#include <string>
#include "rose.h"
#include "Cxx_Grammar.h"
#include "ContractPrinter.hpp"
#include "RoseHelpers.hpp"


using namespace std;


void
ContractPrinter::visit(
  /* in */ SgNode* node)
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
              printLineComment(node, "Precondition clause:", true);
              cout<<(*iter)->getString()<<endl;
            }
            else if (str.find("ENSURE")!=string::npos)
            {
              printLineComment(node, "Postcondition clause:", true);
              cout<<(*iter)->getString()<<endl;
            }
            else if (str.find("INVARIANT")!=string::npos)
            {
              printLineComment(node, "Invariant clause:", true);
              cout<<(*iter)->getString()<<endl;
            }
            else if (str.find("ASSERT")!=string::npos)
            {
              printLineComment(node, "Assertion clause:", true);
              cout<<(*iter)->getString()<<endl;
            }
            else if (str.find("INIT")!=string::npos)
            {
              printLineComment(node, "Initialization:", true);
              cout<<(*iter)->getString()<<endl;
            }
            else if (str.find("FINAL")!=string::npos)
            {
              printLineComment(node, "Finalization:", true);
              cout<<(*iter)->getString()<<endl;
            }
            else
            {
              printLineComment(node, 
                "WARNING: Unidentified contract annotation:", true);
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

