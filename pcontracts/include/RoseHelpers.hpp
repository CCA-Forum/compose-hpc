/**
 * \internal
 * File:           RoseHelpers.hpp
 * Author:         T. Dahlgren
 * Created:        2012 August 3
 * Last Modified:  2013 August 2
 * \endinternal
 *
 *
 * @file
 * @brief
 * Helper or utility routines related to ROSE features.
 *
 * @todo Should this be encapsulated in a class?
 *
 * @htmlinclude copyright.html
 */

#ifndef include_Rose_Helpers_hpp
#define include_Rose_Helpers_hpp

#include <string.h>
#include "rose.h"

#define FILE_INFO Sg_File_Info::generateDefaultFileInfoForTransformationNode()


/**
 * Attach a translation comment to the node.
 *
 * @param[in,out]  node  Current AST node.
 * @param[in]      cmt   The comment to be attached to the node.
 */
void 
attachTranslationComment(SgNode* node, std::string cmt);


/**
 * Get the basic signature.  
 *
 * @param[in]  decl  The function declaration.
 * @return           A basic signature derived from the node's unparsed output.
 */
std::string
getBasicSignature(SgFunctionDeclaration* decl);


/**
 * Get the basic signature.  
 *
 * @param[in]  def  The function definition.
 * @return          A basic signature derived from the node's unparsed output.
 */
std::string
getBasicSignature(const SgFunctionDefinition* def);


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
 * @param[in] lang  The output language.
 * @return          The name of the language option.
 */
std::string
getLanguageOptionName(SgFile::outputLanguageOption_enum lang);


/**
 * Add the statement to each return point of the given function.
 *
 * Source:  
 * This function is a slight variant of SageInterface::instrumentEndOfFunction
 * provided in ROSE. (2014 June 6)
 *
 * @param[in] decl   The declaration of the function of interest.
 * @param[in] sttmt  The statement to be added prior to each return point.
 * @return           The number of instrumented return points.
 */
int
instrumentReturnPoints(SgFunctionDeclaration* decl, SgStatement* sttmt);


/**
 * Determines if the specified directive type is a C/C++ comment.
 *
 * @param[in] dType  The type of preprocessing directive.
 * @return           True if the directive is a C/C++ comment; otherwise, false.
 */
bool
isCComment(PreprocessingInfo::DirectiveType dType);


/**
 * Determine if the named file is in the input file list.  
 * 
 * Using this check SEEMS to be the most reliable way of ensuring functions
 * defined elsewhere are not instrumented.
 *
 * @param[in] project   Sage project/AST.
 * @param[in] filename  Name of the file to be checked.
 * @return              True if filename is in the list; otherwise, false.
 */
bool
isInputFile(SgProject* project, std::string filename);


/**
 * Prints the specified comment followed by the node type, line, and file
 * information of the associated node.
 *
 * @param[in] node     The associated AST node.
 * @param[in] cmt      The comment to be printed.
 * @param[in] addFile  Include the file name in the message.
 */
void
printLineComment(SgNode* node, std::string cmt, bool addFile);

/**
 * Strip and compress white space.  Strip leading and trailing white space
 * and replace and compress embedded white space to at most one blank between
 * non-white space contents.
 *
 * @param[in] txt  The text to be cleaned.
 * @return         The cleaned up version of the text.
 */
std::string
compress(std::string txt);


#endif /* include_Rose_Helpers_hpp */
