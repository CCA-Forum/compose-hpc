/*
 * File:         PaulContractsDecorate.h
 * Description:  TEMPORARY Paul interface contracts decorate.
 * Source:       Just the decorate part of PaulDecorate.cpp (but renamed).
 */

#ifndef included_PaulContractsDecorate_h
#define included_PaulContractsDecorate_h

#include "rose.h"

/**
 * This function is used to decorate a SgProject with parsed
 * structured comments treated as AstAttributes.  The
 * SgProject parameter must be initialized using the usual methods
 * for creating an SgProject in ROSE.  The second parameter specifies
 * the configuration file that contains the tag->parser mappings used
 * by PAUL to determine which parser to invoke for each tag that it
 * encounters.
 *
 * \param sageProject The SgProject to perform the decoration traversal on.
 * \param conf_fname The filename of the PAUL configuration file.
 */
void paulContractsDecorate (SgProject* sageProject, string conf_fname) ;

#endif /* included_PaulContractsDecorate_h */

