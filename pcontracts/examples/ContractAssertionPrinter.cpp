/**
 * File:          ContractAssertionPrinter.cpp
 * Author:        T. Dahlgren
 * Created:       2012 July 6
 * Last Modified: 2012 July 6
 *
 * @file
 * @section DESCRIPTION
 * Simple class implementation for illustrating the use of basic ROSE features 
 * for printing contract clause assertions.  It is NOT intended for contract
 * enforcement use.
 *
 * @section LICENSE
 * TBD
 *
 * @todo Clean up this example so it can "properly" re-use ContractPrinter.
 *
 * @todo Need to "properly" dispose of any trailing comment marks.
 *
 * @todo 
 */

#include <iostream>
#include <string>
#include "rose.h"
#include "Cxx_Grammar.h"
#include "ContractAssertionPrinter.hpp"
#include "ContractPrinter.hpp"

using namespace std;


/**
 * Print individual assertion.
 *
 * @param statement  The contract clause assertion.
 *
 * @todo Get rid of extra whitespace.
 */
void
printStatement(string statement)
{
  if (!statement.empty())
  {
    int i = 0;
    string labels[] = { "label/error comment", "assertion" };

    char* str = new char [statement.size()+1]; 
    strcpy(str, statement.c_str());

    char* ptr = strtok(str, ":");
    while (ptr != NULL)
    {
      if (i<=1)
      {
        cout<<"  "<<labels[i++]<<": ";
      }
      cout<<ptr<<endl;
      ptr = strtok(NULL, ":");
    }

    delete [] str;
  }

  return;
} /* printStatement */

/**
 * Print individual contract clause assertions.
 *
 * @param clause  The contract clause text extracted from the structured 
 *                  comment.
 *
 * @todo Properly remove comment marks.
 */
void
printAssertions(string clause)
{
  char* ptr;

  if (!clause.empty())
  {
    char* cstr = new char [clause.size()+1]; 
    strcpy(cstr, clause.c_str());

    ptr = strtok(cstr, ";");
    while (ptr != NULL)
    {
      printStatement(ptr);
      ptr = strtok(NULL, ";");
    }
    cout<<endl;

    delete [] cstr;
  }

  return;
} /* printAssertions */


void
ContractAssertionPrinter::visit(SgNode* node)
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
        if (isComment(((*iter)->getTypeOfDirective())))
        {
          size_t pos;
          string str = (*iter)->getString();
          if ((pos=str.find("CONTRACT"))!=string::npos)
          {
            if ((pos=str.find("REQUIRE"))!=string::npos)
            {
              printLineComment(node, "Precondition clause:");
	      printAssertions((*iter)->getString().substr(pos+7));
            }
            else if ((pos=str.find("ENSURE"))!=string::npos)
            {
              printLineComment(node, "Postcondition clause:");
	      printAssertions((*iter)->getString().substr(pos+6));
            }
            else if ((pos=str.find("INVARIANT"))!=string::npos)
            {
              printLineComment(node, "Invariant clause:");
	      printAssertions((*iter)->getString().substr(pos+9));
            }
            else
            {
              printLineComment(node, "WARNING: Unidentified contract clause:");
	      printAssertions((*iter)->getString().substr(pos+8));
            }
          }
        }
      }
      cout<<endl;
    }

    /* Do NOT attempt to delete lNode or cmts! */
  }
  return;
} /* ContractAssertionPrinter::visit */

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
    ContractAssertionPrinter* vis = new ContractAssertionPrinter();

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

