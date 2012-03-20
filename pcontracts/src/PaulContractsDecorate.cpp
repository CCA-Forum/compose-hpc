/*
 * File:         PaulContractsDecorate.cpp
 * Description:  TEMPORARY Paul interface contracts decorate.
 * Source:       Just the decorate part of PaulDecorate.cpp (but renamed).
 *               
 */
#include <iostream>
#include "rose.h"
#include "PaulConfReader.h"
//#include "PaulDecorate.h"
#include "PaulContractsDecorator.h"
#include "CommentVisitor.hpp"

void paulContractsDecorate(SgProject* project, string conf_fname)
{
  CommentVisitor v;

  if (conf_fname != "") {
    paul_tag_map ptm = read_paul_conf(conf_fname);
    v.setTagMap(ptm);
  }

  v.traverseInputFiles(project,preorder);    // FIXME:?
}
