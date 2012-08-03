/**
 * File:          ContractAssertionPrinter.cpp
 * Author:        T. Dahlgren
 * Created:       2012 July 6
 * Last Modified: 2012 August 03
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
 * @todo  Spend some time determining why multiple assertion expressions are NOT
 *   being printed when a label is involved.  ((Bad use of strtok!!!))
 *
 * @todo  Make sure the output matches what is expected from the input.
 */

#include <iostream>
#include <string>
#include "rose.h"
#include "Cxx_Grammar.h"
#include "ContractAssertionPrinter.hpp"
#include "ContractPrinter.hpp"

using namespace std;

string
removeWS(string part)
{
  if (!part.empty())
  {
    int i;
    for (i=0; i<part.length(); i++)
    {
      if ( (part[i] == '\t') || (part[i] == '\n') )
      {
        part[i] = ' ';
      }
    }
  
    size_t start = 0, end;
    while ( (end=part.find("  ", start)) != string::npos )
    {
      part.replace(end, 2, 1, ' ');
      start = end;
    }

    start=part.find_first_not_of(' ');
    end=part.find_last_not_of(' ');
    if ( (start != string::npos) && (end != string::npos) )
    {
      part=part.substr(start, end-start+1);
    }
  }

  return part;
} /* removeWS */


/**
 * Print each assertion expression within the clause.
 *
 * @param clause  The contract clause text extracted from the structured 
 *                  comment.
 */
void
printClause(string clause)
{
  string labels[] = { 
   "label/error comment ", 
   "assertion expression" 
  };

  if (!clause.empty())
  {
    size_t startAE = 0, endAE;
    while ( (endAE=clause.find(";", startAE)) != string::npos )
    {
      cout<<"\n";
      string statement = clause.substr(startAE, endAE-startAE);
      if (!statement.empty())
      {
        string label, expr;
        size_t startE = 0, endL;
        if ( (endL=statement.find(":")) != string::npos )
        {
          label = removeWS(statement.substr(0, endL));
          startE = endL+1;
          cout<<"   "<<labels[0]<<": "<<label<<endl;
        }
    
        expr = removeWS(statement.substr(startE));
        cout<<"   "<<labels[1]<<": "<<expr<<endl;
      }

      startAE = endAE+1;
    }
  }

  return;
} /* printClause */


/**
 * Process the comment, processing encountered contract clauses.
 *
 * @param cmt  Comment contents.
 */
void
processCommentContents(SgNode* node, const string cmt)
{
  if ( (node != NULL) && !cmt.empty() )
  {
    size_t pos;
    if ((pos=cmt.find("CONTRACT"))!=string::npos)
    {
      if ((pos=cmt.find("REQUIRE"))!=string::npos)
      {
        printLineComment(node, "Precondition clause:");
        printClause(cmt.substr(pos+7));
      }
      else if ((pos=cmt.find("ENSURE"))!=string::npos)
      {
        printLineComment(node, "Postcondition clause:");
        printClause(cmt.substr(pos+6));
      }
      else if ((pos=cmt.find("INVARIANT"))!=string::npos)
      {
        printLineComment(node, "Invariant clause:");
        printClause(cmt.substr(pos+9));
      }
      else
      {
        printLineComment(node, "WARNING: Unidentified contract clause:");
        printClause(cmt.substr(pos+8));
      }
    }
  }
  return;
} /* processCommentContents */


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
          case PreprocessingInfo::FortranStyleComment:
          case PreprocessingInfo::F90StyleComment:
            {
              string str = (*iter)->getString();
              processCommentContents(node, str.substr(1));
            }
            break;
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

