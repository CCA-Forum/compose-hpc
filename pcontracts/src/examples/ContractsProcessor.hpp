/**
 * \internal
 * File:           ContractsProcessor.hpp
 * Author:         T. Dahlgren
 * Created:        2012 November 1
 * Last Modified:  2013 March 26
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


using namespace std;

/**
 * Contract enforcement include files.
 */
const string S_INCLUDE_BASICS   = "#include \"contracts.h\"";
const string S_INCLUDE_ENFORCER = "#include \"ContractsEnforcer.h\"";
const string S_INCLUDE_OPTIONS  = "#include \"contractOptions.h\"";
const string S_INCLUDES = S_INCLUDE_BASICS + "\n" + S_INCLUDE_ENFORCER + "\n"
                           + S_INCLUDE_OPTIONS;

class ContractsProcessor
{
  public:
    ContractsProcessor() : d_invariants(NULL) {}
    ~ContractsProcessor() { if (d_invariants != NULL) delete d_invariants; }

    void addExpressions(string clause, ContractComment* cc, 
      bool firstExecClause);
    
    int addFinalize(SgFunctionDefinition* def, SgBasicBlock* body, 
      ContractComment* cc);
    
    int addIncludes(SgGlobal* globalScope);
    
    int addIncludes(SgProject* project, bool skipTransforms);
    
    int addInitialize(SgBasicBlock* body, ContractComment* cc);
    
    int addPostChecks(SgFunctionDefinition* def, SgBasicBlock* body, 
      ContractComment* cc);
    
    int addPreChecks(SgBasicBlock* body, ContractComment* cc);

    SgExprStatement* buildCheck(SgStatement* scope, 
      ContractClauseEnum clauseType, AssertionExpression ae, 
      PPIDirectiveType dt);
    
    void extractContract(SgLocatedNode* lNode, bool firstExec, 
      ContractClauseType &clause);
    
    ContractComment* extractContractComment(SgNode* aNode, 
      AttachedPreprocessingInfoType::iterator info, bool firstExecClause);
    
    bool inInvariants(string nm);
    
    int instrumentRoutines(SgProject* project, bool skipTransforms);
    
    bool isExecutable(string expr);
    
    ContractComment* processCommentEntry(SgNode* aNode, string cmt, 
      PreprocessingInfo::DirectiveType dirType, bool firstExecClause);
    
    int processFunctionComments(SgFunctionDefinition* def);

    int processFunctionDef(SgFunctionDefinition* def);

    int processNonFunctionNode(SgLocatedNode* lNode);

  private:
    ContractComment*  d_invariants;

    /** Global first (routine) clause flag. */
    bool  d_first;

    void setInvariants(SgLocatedNode* lNode, ContractComment* cc);

    int processAssert(SgLocatedNode* lNode, ContractComment* cc);

    int processInit(SgLocatedNode* lNode, ContractComment* cc);

    int processFinal(SgLocatedNode* lNode, ContractComment* cc);

};  /* class ContractsProcessor */

#endif /* include_Contracts_Processor_hpp */
