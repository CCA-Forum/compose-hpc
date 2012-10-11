/**
 * File:           RoseHelpers.cpp
 * Author:         T. Dahlgren
 * Created:        2012 August 3
 * Last Modified:  2012 October 11
 *
 * @file
 * @section DESCRIPTION
 * Helper or utility routines related to ROSE features.
 *
 * @section LICENSE
 * TBD
 */

#include <iostream>
#include <string>
#include "rose.h"
#include "RoseHelpers.hpp"

using namespace std;


void
attachTranslationComment(SgNode* node, std::string cmt)
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
getBasicSignature(SgFunctionDeclaration* decl)
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
getLanguageOptionName(
SgFile::outputLanguageOption_enum lang)
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


bool
isCComment(PreprocessingInfo::DirectiveType dType)
{
  return (  (dType == PreprocessingInfo::C_StyleComment)
         || (dType == PreprocessingInfo::CplusplusStyleComment)  );
} /* isCComment */


bool
isInputFile(SgProject* project, string filename)
{
  bool isIF = false;

  SgFilePtrList::iterator iter;
  for (iter=project->get_fileList().begin();
       iter!=project->get_fileList().end() && !isIF; iter++)
  {
    if (filename == (((*iter)->get_sourceFileNameWithPath())))
      isIF = true;
  }

  return isIF;
}  /* isInputFile */


void
printLineComment(SgNode* node, string cmt)
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


string
removeWS(string txt)
{
  if (!txt.empty())
  {
    int i;
    for (i=0; i<txt.length(); i++)
    {
      if ( (txt[i] == '\t') || (txt[i] == '\n') || (txt[i] == '\r') )
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
} /* removeWS */
