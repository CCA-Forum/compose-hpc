/**
 * \internal
 * File:           ContractsProcessor.hpp
 * Author:         T. Dahlgren
 * Created:        2012 November 1
 * Last Modified:  2012 November 28
 * \endinternal
 *
 * @file
 * @brief
 * Basic contract clause processing utilities.
 *
 * @htmlinclude copyright.html
 */

#ifndef include_Contracts_Processor_hpp
#define include_Contracts_Processor_hpp

#include <list>
#include <string>
#include "rose.h"
#include "AssertionExpression.hpp"
#include "ContractComment.hpp"


using namespace std;


class ContractsProcessor
{
  public:
    ContractsProcessor() : d_invariants(NULL) {}
    ~ContractsProcessor() { if (d_invariants != NULL) delete d_invariants; }

   int addPreChecks(SgBasicBlock* body, ContractComment* cc);

    int addPostChecks(SgFunctionDefinition* def, SgBasicBlock* body, 
      ContractComment* cc);
    
    void addExpressions(string clause, ContractComment* cc, 
      bool firstExecClause);
    
    int
    addFinalize(SgFunctionDefinition* def, SgBasicBlock* body, 
      ContractComment* cc);
    
    int addIncludes(SgProject* project, bool skipTransforms);
    
    int addInitialize(SgBasicBlock* body, ContractComment* cc);
    
    SgExprStatement* buildCheck(SgBasicBlock* body, 
      ContractClauseEnum clauseType, AssertionExpression ae, 
      PPIDirectiveType dt);
    
    ContractComment* extractContractClause(SgFunctionDeclaration* decl, 
      AttachedPreprocessingInfoType::iterator info, bool firstExecClause);
    
    bool inInvariants(string nm);
    
    int instrumentRoutines(SgProject* project, bool skipTransforms);
    
    bool isExecutable(string expr);
    
    ContractComment* processCommentEntry(SgFunctionDeclaration* dNode, 
      string cmt, PreprocessingInfo::DirectiveType dirType, 
      bool firstExecClause);
    
    int processComments(SgFunctionDefinition* def);

  private:
    ContractComment*  d_invariants;

};  /* class ContractsProcessor */

#endif /* include_Contracts_Processor_hpp */
