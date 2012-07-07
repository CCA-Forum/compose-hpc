/**
 * File:          ContractPrinter.cpp
 * Author:        T. Dahlgren
 * Created:       2012 July 6
 * Last Modified: 2012 July 6
 *
 * @file
 * @section DESCRIPTION
 * Simple class implementation for illustrating the use of basic ROSE features 
 * for printing contract clause comments.  It is NOT intended for contract
 * enforcement use.
 *
 * @section LICENSE
 * TBD
 *
 * @todo Clean up this example so this class can be re-used "properly"
 * in ContractAssertionPrinter.cpp.
 */

#include <iostream>
#include <string>
#include "rose.h"
#include "Cxx_Grammar.h"
#include "ContractPrinter.hpp"

using namespace std;


bool
ContractPrinter::isComment(PreprocessingInfo::DirectiveType dType)
{
  return (  (dType == PreprocessingInfo::C_StyleComment)
         || (dType == PreprocessingInfo::CplusplusStyleComment)  );
} /* isComment */


void
ContractPrinter::printLineComment(SgNode* node, const char* cmt)
{
  SgLocatedNode* lNode = isSgLocatedNode(node);
  if (lNode != NULL)
  {
    Sg_File_Info* info = lNode->get_file_info();
    if (info != NULL)
    {
      cout<<"\n"<<cmt<<"\n   @line "<<info->get_raw_line();
      cout<<" of "<<info->get_raw_filename()<<endl;
    }
  }

  return;
} /* printLineComment */


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
        if (isComment(((*iter)->getTypeOfDirective())))
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
            else
            {
              printLineComment(node, "WARNING: Unidentified contract clause:");
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

