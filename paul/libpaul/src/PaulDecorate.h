#ifndef PAULDECORATE_H
#define PAULDECORATE_H

#include <string>
#include "rose.h"

using namespace std;

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
void paulDecorate (SgProject* sageProject, string conf_fname) ;

#endif

