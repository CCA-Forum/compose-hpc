/**
 * \internal
 * File:          ContractAssertionPrinter.cpp
 * Author:        T. Dahlgren
 * Created:       2012 July 6
 * Last Modified: 2013 September 20
 * \endinternal
 *
 * @file
 * @brief 
 * Class for printing contract clause assertions detected in ROSE AST comments.
 *
 * @details
 * Simple class implementation for illustrating the use of basic ROSE features 
 * and a rudimentary contract clause parser for extracting and printing 
 * contract clause assertions.  
 *
 * @htmlinclude copyright.html
 */

#include <iostream>
#include <string>
#include "rose.h"
#include "Cxx_Grammar.h"
#include "ContractAssertionPrinter.hpp"
#include "ContractPrinter.hpp"
#include "RoseHelpers.hpp"

using namespace std;


/**
 * Print each assertion expression within the clause.
 *
 * @param[in] clause  The contract clause text extracted from the structured 
 *                      comment.
 */
void
printClause(
 /* in */ string clause)
{
  if (!clause.empty())
  {
    size_t startAE = 0, endAE;
    while ( (endAE=clause.find(";", startAE)) != string::npos )
    {
      cout<<"\n";
      string statement = clause.substr(startAE, endAE-startAE);
      if (!statement.empty())
      {
        size_t startE = 0, endL;

        string labels[] = { 
         "label/error comment ", 
         "assertion expression" 
        };

        if ( (endL=statement.find(":")) != string::npos )
        {
          string label = compress(statement.substr(0, endL));
          startE = endL+1;
          cout<<"   "<<labels[0]<<": "<<label<<endl;
        }
    
        string expr = compress(statement.substr(startE));
        cout<<"   "<<labels[1]<<": "<<expr<<endl;
      }

      startAE = endAE+1;
    }
  }

  return;
} /* printClause */


/**
 * Process the comment to assess and handle any contract clause.
 *
 * @param[in] node  Current AST node.
 * @param[in] cmt   Comment contents.
 */
void
processCommentContents(
  /* in */ SgNode* node, 
  /* in */ const string cmt)
{
  if ( (node != NULL) && !cmt.empty() )
  {
    size_t pos;
    if ((pos=cmt.find("CONTRACT"))!=string::npos)
    {
      if ((pos=cmt.find("REQUIRE"))!=string::npos)
      {
        printLineComment(node, "Precondition clause:", true);
        printClause(cmt.substr(pos+7));
      }
      else if ((pos=cmt.find("ENSURE"))!=string::npos)
      {
        printLineComment(node, "Postcondition clause:", true);
        printClause(cmt.substr(pos+6));
      }
      else if ((pos=cmt.find("INVARIANT"))!=string::npos)
      {
        printLineComment(node, "Invariant clause:", true);
        printClause(cmt.substr(pos+9));
      }
      else if ((pos=cmt.find("ASSERT"))!=string::npos)
      {
        printLineComment(node, "Assertion clause:", true);
        printClause(cmt.substr(pos+6));
      }
      else if ((pos=cmt.find("INIT"))!=string::npos)
      {
        printLineComment(node, "Initialization:", true);
        printClause(cmt.substr(pos+4));
      }
      else if ((pos=cmt.find("FINAL"))!=string::npos)
      {
        printLineComment(node, "Finalization:", true);
        printClause(cmt.substr(pos+5));
      }
      else if ((pos=cmt.find("STATS"))!=string::npos)
      {
        printLineComment(node, "Finalization:", true);
        printClause(cmt.substr(pos+5));
      }
      else
      {
        printLineComment(node, "WARNING: Unidentified contract annotation:",
                         true);
        printClause(cmt.substr(pos));
      }
    }
  }
  return;
} /* processCommentContents */


/**
 * Process the current AST node passed by the front end, identifying
 * and printing individual contract clause assertions.
 *
 * @param[in] node  Current AST node.
 */
void
ContractAssertionPrinter::visit(
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
