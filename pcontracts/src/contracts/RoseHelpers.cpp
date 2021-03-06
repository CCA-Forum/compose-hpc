/**
 * \internal
 * File:           RoseHelpers.cpp
 * Author:         T. Dahlgren
 * Created:        2012 August 3
 * Last Modified:  2015 June 25
 * \endinternal
 *
 * @file
 * @brief
 * Helper or utility routines related to ROSE features.
 *
 * @htmlinclude copyright.html
 */

#include <cctype>
#include <iostream>
#include <string>
#include "rose.h"
#include "RoseHelpers.hpp"


using namespace std;


void
attachTranslationComment(
  /* inout */ SgNode*     node, 
  /* in */    std::string cmt)
{
  SgLocatedNode* lNode = isSgLocatedNode(node);
  if ( (lNode != NULL) && !cmt.empty() )
  {
#ifdef DEBUG
    cout<<"DEBUG: ...attaching comment ("<<cmt<<")...\n";
#endif /* DEBUG */

    SageInterface::attachComment(lNode, cmt);
  }
} /* attachTranslationComment */


SgFile::outputLanguageOption_enum
getCurrentLanguageOption()
{
  SgFile::outputLanguageOption_enum lang = SgFile::e_error_output_language;

  if (SageInterface::is_C_language())
  {
    lang = SgFile::e_C_output_language;
  }
  else if (SageInterface::is_Cxx_language())
  {
    lang = SgFile::e_Cxx_output_language;
  }
  else if (SageInterface::is_Fortran_language())
  {
    lang = SgFile::e_Fortran_output_language;
  }
  else if (SageInterface::is_Java_language())
  {
    lang = SgFile::e_Java_output_language;
  }
/*
 * No equivalent in SageInterface:
 *
  else if (SageInterface::is_Promela_language())
  {
    lang = SgFile::e_Promela_output_language;
  }
 */
  else if (SageInterface::is_PHP_language())
  {
    lang = SgFile::e_PHP_output_language;
  }
  else if (SageInterface::is_Python_language())
  {
    lang = SgFile::e_Python_output_language;
  }
 
  return lang;
}  /* getCurrentLanguageOption */


string
getBasicSignature(
  /* in */ SgFunctionDeclaration* decl)
{
  string res;

  if (decl != NULL)
  {
    string declStr = decl->unparseToString();
    size_t bst = declStr.find_first_of("{");
    if (bst!=string::npos)
    {
      res.append(declStr.substr(0, bst));
      res.append(";");
    }
    else
    {
      cerr<<"\nERROR:  Failed to locate starting (body) brace: "<<decl<<endl;
    }
  }

  return res;
}  /* getBasicSignature */


string
getBasicSignature(
  /* in */ const SgFunctionDefinition* def)
{
  string res;

  if (def != NULL)
  {
    SgFunctionDeclaration* decl = def->get_declaration();
    if (decl != NULL)
    {
      res = getBasicSignature(decl);
    }
    else
    {
      cerr<<"\nERROR:  getBasicSignature requires function declaration.\n";
    }
  }
  else
  {
    cerr<<"\nERROR:  getBasicSignature requires function definition.\n";
  }

  return res;
}  /* getBasicSignature */


string
getLanguageOptionName(
  /* in */ SgFile::outputLanguageOption_enum lang)
{
  string res;

  switch (lang)
  {
    case SgFile::e_error_output_language:
      {
        res = "Error";
      }
      break;
    case SgFile::e_default_output_language:
      {
        res = "Default";
      }
      break;
    case SgFile::e_C_output_language:
      {
        res = "C";
      }
      break;
    case SgFile::e_Cxx_output_language:
      {
        res = "Cxx";
      }
      break;
    case SgFile::e_Fortran_output_language:
      {
        res = "Fortran";
      }
      break;
    case SgFile::e_Java_output_language:
      {
        res = "Java";
      }
      break;
    case SgFile::e_Promela_output_language:
      {
        res = "Promela";
      }
      break;
    case SgFile::e_PHP_output_language:
      {
        res = "PHP";
      }
      break;
    case SgFile::e_Python_output_language:
      {
        res = "Python";
      }
      break;
    default:
      {
        res = "Unrecognized";
      }
      break;
  }

  return res;
} /* getLanguageOptionName */


int
instrumentReturnPoints(SgFunctionDeclaration* decl, 
  std::vector<SgStatement*> stmtList, std::string resName)
{
  int result = 0;
  int numStmts = stmtList.size();

  if ( (decl != NULL) && (numStmts > 0) )
  {
    // The following code is a slightly modified variant of ROSE's
    // SageInterface::instrumentEndOfFunction, which did not check
    // whether the function had a return type or not before performing
    // the rewrite. (2014 June 6)
    //
    // The code has since been extended to ensure the result variable
    // of a post-condition clause has the right value prior to the]
    // contract check.
    Rose_STL_Container<SgNode*> stmts = NodeQuery::querySubTree(decl, 
        V_SgReturnStmt);

    Rose_STL_Container<SgNode*>::iterator i;
    for (i = stmts.begin(); i != stmts.end(); i++)
    {
      SgReturnStmt* currStmt = isSgReturnStmt(*i);
      if (currStmt != NULL)
      {
        SgExpression* exp = currStmt->get_expression();

        // TV (05/03/2011) Catch the case "return ;" where exp is NULL
        // TLD (08/06/2015) At this point it should only matter for rewrite-
        //     purposes if this is a function.
        if (!isSgTypeVoid(decl->get_type()->get_return_type()))
        {
          SageInterface::splitExpression(exp, resName);
        }

        for (std::vector<SgStatement*>::iterator iter = stmtList.begin();
             iter != stmtList.end(); iter++)
        {
          SgStatement* sttmt;

          // avoid reusing the statement
          if (result >= numStmts) 
          {
              sttmt = SageInterface::copyStatement(*iter);
          } 
          else
          { 
              sttmt = *iter;
          } 
          sttmt->set_parent(currStmt->get_parent());

          SageInterface::insertStatementBefore(currStmt, sttmt);
          result++;
        }
      }
    } // for each return statement

    if (stmts.size() == 0) // a function without any return at all,
    {
      SgBasicBlock * body = decl->get_definition()->get_body();
      if (body == NULL)
      {
        cerr<<"In instrumentReturnPoints(), ";
        cerr<<"found a missing function body!\n";
        ROSE_ASSERT(false);
      }

      for (std::vector<SgStatement*>::iterator iter = stmtList.begin();
           iter != stmtList.end(); iter++)
      {
        SageInterface::appendStatement((*iter), body);
        result++;
      }
    }
  }

  return result;
} /* instrumentReturnPoints */


bool
isCComment(
  /* in */ PreprocessingInfo::DirectiveType dType)
{
  return (  (dType == PreprocessingInfo::C_StyleComment)
         || (dType == PreprocessingInfo::CplusplusStyleComment)  );
} /* isCComment */


bool
isInputFile(
  /* in */ SgProject* project, 
  /* in */ string     filename)
{
  bool isIF = false;

  SgFilePtrList::iterator iter;
  for (iter=project->get_fileList().begin();
       iter!=project->get_fileList().end() && !isIF; iter++)
  {
    if (filename == (((*iter)->get_sourceFileNameWithPath())))
    {
      isIF = true;
    }
  }

  return isIF;
}  /* isInputFile */


void
printLineComment(
  /* in */ SgNode* node, 
  /* in */ string  cmt,
  /* in */ bool    addFile)
{
  SgLocatedNode* lNode = isSgLocatedNode(node);
  if (lNode != NULL)
  {
    Sg_File_Info* info = lNode->get_file_info();
    if (info != NULL)
    {
      if (addFile)
      {
        cout<<"\n"<<cmt<<"\n   @line "<<info->get_raw_line();
        cout<<" of "<<info->get_raw_filename();
      }
      else
      {
        cout<<cmt<<" (@line "<<info->get_raw_line()<<")";
      }
      cout<<endl;
    }
  }

  return;
} /* printLineComment */


string
compress(
  /* in */ string txt)
{
  if (!txt.empty())
  {
    int i;
    for (i=0; i<txt.length(); i++)
    {
      if (isspace(txt[i]) || iscntrl(txt[i]))  // Ensure no extraneous controls
      {
        txt[i] = ' ';
      }
    }

    size_t start = 0, end;
    while ( (end=txt.find("  ", start)) != string::npos )
    {
      txt.replace(end, 2, 1, ' ');
      start = end;
    }

    start=txt.find_first_not_of(' ');
    end=txt.find_last_not_of(' ');
    if ( (start != string::npos) && (end != string::npos) )
    {
      txt=txt.substr(start, end-start+1);
    }
  }

  return txt;
} /* compress */
