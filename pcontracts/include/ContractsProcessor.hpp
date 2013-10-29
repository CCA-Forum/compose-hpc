/**
 * \internal
 * File:           ContractsProcessor.hpp
 * Author:         T. Dahlgren
 * Created:        2012 November 1
 * Last Modified:  2013 October 28
 * \endinternal
 *
 * @file
 * @brief
 * Basic contract clause processing utilities.
 *
 *
 * @todo Consider refactoring to move addIncludes() and instrumentRoutines()
 *  to a subclass used by the routines instrumenter example.
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


/**
 * Contract enforcement include files.
 */
const std::string S_INCLUDE_BASICS      = "#include \"contracts.h\"";
const std::string S_INCLUDE_ENFORCER    = "#include \"ContractsEnforcer.h\"";
const std::string S_INCLUDE_EXPRESSIONS = "#include \"ExpressionRoutines.h\"";
const std::string S_INCLUDE_OPTIONS     = "#include \"contractOptions.h\"";
const std::string S_INCLUDES = S_INCLUDE_BASICS + "\n" + S_INCLUDE_ENFORCER 
          + "\n" + S_INCLUDE_OPTIONS + "\n" + S_INCLUDE_EXPRESSIONS;

class ContractsProcessor
{
  public:
    ContractsProcessor() : d_invariants(NULL), d_first(false) {}
    ~ContractsProcessor() { if (d_invariants != NULL) {delete d_invariants;} }

    void addExpressions(const std::string clause, ContractComment* cc);
    
    int addFinalize(const SgFunctionDefinition* def, const SgBasicBlock* body, 
      ContractComment* cc);
    
    int addIncludes(SgGlobal* globalScope);
    
    int addIncludes(SgProject* project, bool skipTransforms);
    
    int addInitialize(SgBasicBlock* body, ContractComment* cc);
    
    int addPostChecks(const SgFunctionDefinition* def, SgBasicBlock* body, 
      ContractComment* cc);
    
    int addPreChecks(SgBasicBlock* body, ContractComment* cc);

    int addStatsDump(const SgFunctionDefinition* def, SgBasicBlock* body,
      ContractComment* cc);

    int addTimeUpdate(SgBasicBlock* body);

    int addReturnVariable(SgBasicBlock* body, ContractComment* cc, 
      SgType* returnType);

    SgExprStatement* buildCheck(const SgStatement* currSttmt, 
      ContractClauseEnum clauseType, AssertionExpression ae, 
      PPIDirectiveType dt);
    
    void extractContract(SgLocatedNode* lNode, ContractClauseType &clauses);
    
    ContractComment* extractContractComment(SgNode* aNode, 
      AttachedPreprocessingInfoType::iterator info);
    
    bool hasContractComment(SgLocatedNode* lNode);
    
    bool inInvariants(std::string nm);
    
    int instrumentRoutines(SgProject* project, bool skipTransforms);
    
    bool isExecutable(std::string expr);
    
    ContractComment* processCommentEntry(SgNode* aNode, std::string cmt, 
      PreprocessingInfo::DirectiveType dirType);
    
    int processFunctionComments(SgFunctionDefinition* def);

    int processFunctionDef(SgFunctionDefinition* def);

    int processNonFunctionNode(SgLocatedNode* lNode);

  private:
    ContractComment*  d_invariants;

    /** Global first (routine) clause flag. */
    bool  d_first;

    ContractsProcessor& operator=( const ContractsProcessor&);

    ContractsProcessor(ContractsProcessor&);

    void setInvariants(SgLocatedNode* lNode, ContractComment* cc);

    int processAssert(SgLocatedNode* lNode, ContractComment* cc);

    int processFinal(SgLocatedNode* lNode, ContractComment* cc);

    int processInit(SgLocatedNode* lNode, ContractComment* cc);

    int processStats(SgLocatedNode* lNode, ContractComment* cc);

};  /* class ContractsProcessor */

#endif /* include_Contracts_Processor_hpp */
