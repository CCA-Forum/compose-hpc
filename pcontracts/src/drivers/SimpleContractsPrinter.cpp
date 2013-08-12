/**
 * \internal
 * File:  SimpleContractsPrinter.cpp
 * \endinternal
 *
 * @file
 * @brief
 * Simple contracts visitor class for printing PAUL CONTRACT annotations.
 *
 * @htmlinclude copyright.html
 */
//#include "PaulDecorate.h"
//#include "KVAnnotationValue.h"
#include "PaulContractsCommon.h"
#include "rose.h"
#include "SimpleContractsPrinter.hpp"

/**
 * Visit the node to determine if there are any contract annotations.
 * If so, then print them out.
 *
 * @param[in]  node  Current ASST node.
 */
void 
SimpleContractsPrinter::visit(
  /* in */  SgNode *node) 
{
  Annotation *annot = (Annotation *)node->getAttribute(L_CONTRACTS_TAG);

  if (annot != NULL) {
    // 1. Retrieve value
    //KVAnnotationValue *val = (KVAnnotationValue *)annot->getValue();
    // 2. Make sure it is a KV annotation value
    //val = isKVAnnotationValue(val);
    //ROSE_ASSERT(val != NULL);

    // 1. Retrieve the contract clause information
    // 2. make sure it is a contract clause
    std::cout << "Found annotated node:" << node->class_name() << std::endl;
    //val->print();
    std::cout << "\n";
  }
} /* visit */
