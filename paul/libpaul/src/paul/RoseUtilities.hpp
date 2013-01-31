/**
 * File:           RoseUtilities.hpp
 * Author:         T. Dahlgren
 * Created:        2012 October 11
 * Last Modified:  2012 October 11
 *
 * @file
 * @section DESCRIPTION
 * Utility routines related to ROSE features.
 *
 * @section SOURCE
 * Fork of Paul Contract's RoseHelpers.
 *
 * @section LICENSE
 * TBD
 */

#ifndef include_Rose_Utilities_hpp
#define include_Rose_Utilities_hpp

#include <iostream>
#include <string.h>
#include "rose.h"

/**
 * Attach a translation comment to the node.
 *
 * @param node  Current AST node.
 * @param cmt   The comment to be attached to the node.
 */
void 
attachTranslationComment(SgNode* node, std::string cmt);


/**
 * Get the basic signature.  
 *
 * @param decl  The function declaration.
 * @return      A basic signature derived from the node's unparsed output.
 */
std::string
getBasicSignature(SgFunctionDeclaration* decl);


/**
 * Determine the (basic) language option for the current AST according
 * to SageInterface.
 *
 * @return  The current output language option.
 */
SgFile::outputLanguageOption_enum
getCurrentLanguageOption();


/**
 * Return the name of the language option from those known at the
 * time this example was written.
 *
 * @param lang  The output language.
 */
std::string
getLanguageOptionName(SgFile::outputLanguageOption_enum lang);


/**
 * Determines if the specified directive type is a C/C++ comment.
 *
 * @param dType  The type of preprocessing directive.
 * @return       true if the directive is a C/C++ comment; otherwise, false.
 */
bool
isCComment(PreprocessingInfo::DirectiveType dType);


/**
 * Determine if the named file is in the input file list.  
 * 
 * Using this check SEEMS to be the most reliable way of ensuring functions
 * defined elsewhere are not instrumented.
 *
 * @param project   Sage project/AST.
 * @param filename  Name of the file to be checked.
 * @return          true if filename is in the list; otherwise, false.
 */
bool
isInputFile(SgProject* project, std::string filename);


/**
 * Prints the specified comment followed by the node type, line, and file
 * information of the associated node.
 *
 * @param node  The associated AST node.
 * @param cmt   The comment to be printed.
 */
void
printLineComment(SgNode* node, std::string cmt);

#endif /* include_Rose_Utilities_hpp */
