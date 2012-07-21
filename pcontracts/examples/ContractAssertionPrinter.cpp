/**
 * File:          ContractAssertionPrinter.cpp
 * Author:        T. Dahlgren
 * Created:       2012 July 6
 * Last Modified: 2012 July 20
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


/**
 * Extract the substring that does not contain preceeding or trailing 
 * whitespace.
 *
 * @param str  The string to be cleaned up.
 */
string
removePrePostWS(string str)
{
  string res;

  if (!str.empty())
  {
    size_t start, end;
    start=str.find_first_not_of(" ");
    end=str.find_last_not_of(" ");
    if ( (start != string::npos) && (end != string::npos) )
    {
      res = str.substr(start, end-start+1);
    }
  }

  return res;
} /* removePrePostWS */


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
    string labels[] = { 
     "label/error comment", 
     "assertion          " 
    };
    if (statement.find(":") != string::npos)
    {
      char* str = new char [statement.size()+1]; 
      strcpy(str, statement.c_str());
  
      int i = 0;
      char* ptr = strtok(str, ":");
      while (ptr != NULL)
      {
        string baseStr = removePrePostWS(ptr);
        if (!baseStr.empty())
        {
          if (i<=1)
          {
            cout<<"  "<<labels[i++]<<": ";
          }
          cout<<baseStr<<"\n";
        }
        else
        {
          i++;
        }

        ptr = strtok(NULL, ":");
      }
  
      delete [] str;
    }
    else
    {
      string baseStr = removePrePostWS(statement);
      if (!baseStr.empty())
      {
        cout<<"  "<<labels[1]<<": "<<baseStr<<"\n";
      }
    }
  }

  return;
} /* printStatement */


/**
 * Print individual contract clause assertions.
 *
 * @param clause  The contract clause text extracted from the structured 
 *                  comment.
 */
void
printAssertions(string clause)
{
  char* ptr;

  if (!clause.empty())
  {
    /* First get rid of 'pesky' formatting characters. */
    size_t pos;
    string ws[] = { "\n", "\t" };
    list<string> badChars (ws, ws+2);
    list<string>::iterator iter;
    for (iter=badChars.begin(); iter != badChars.end(); iter++)
    {
      while ( (pos = clause.find(*iter)) != string::npos)
      {
        clause[pos] = ' ';
      }
    }

    /* Now break the clause into assertion expressions. */
    char* cstr = new char [clause.size()+1]; 
    strcpy(cstr, clause.c_str());
    ptr = strtok(cstr, ";");
    while (ptr != NULL)
    {
      char* tempStr = new char[strlen(ptr)+1];
      strcpy(tempStr, ptr);
      printStatement(ptr);
      delete [] tempStr;

      ptr = strtok(NULL, ";");
    }

    delete [] cstr;
  }

  return;
} /* printAssertions */


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
        printAssertions(cmt.substr(pos+7));
      }
      else if ((pos=cmt.find("ENSURE"))!=string::npos)
      {
        printLineComment(node, "Postcondition clause:");
        printAssertions(cmt.substr(pos+6));
      }
      else if ((pos=cmt.find("INVARIANT"))!=string::npos)
      {
        printLineComment(node, "Invariant clause:");
        printAssertions(cmt.substr(pos+9));
      }
      else
      {
        printLineComment(node, "WARNING: Unidentified contract clause:");
        printAssertions(cmt.substr(pos+8));
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

