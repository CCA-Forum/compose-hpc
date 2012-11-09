/**
 * File:           ContractsProcessor.hpp
 * Author:         T. Dahlgren
 * Created:        2012 November 1
 * Last Modified:  2012 November 1
 *
 * @file
 * @section DESCRIPTION
 * Basic contract clause processing utilities and supporting classes.
 *
 * @section SOURCE
 * This code was originally part of the initial ContractInstrumenter.cpp,
 * which was renamed to RoutineContractInstrumenter.cpp.
 * 
 * @section LICENSE
 * TBD
 */

#ifndef include_Contracts_Processor_hpp
#define include_Contracts_Processor_hpp

//#include <iostream>
#include <list>
#include <string>
#include "rose.h"
//#include "Cxx_Grammar.h"
//#include "RoseHelpers.hpp"
#include "contractOptions.h"
#include "contractClauseTypes.hpp"

#define FILE_INFO Sg_File_Info::generateDefaultFileInfoForTransformationNode()
#define PPIDirectiveType PreprocessingInfo::DirectiveType

using namespace std;


/*
 *************************************************************************
 * Helper Classes
 *************************************************************************
 */

/**
 * Assertion expression data.
 */
class AssertionExpression 
{
  public:
    AssertionExpression(string l, string expr, AssertionSupportEnum level,
      bool isFirst) 
      : d_label(l), d_expr(expr), d_level(level), d_isFirst(isFirst) {}
    string label() { return d_label; }
    string expr() { return d_expr; }
    AssertionSupportEnum support() { return d_level; }
    bool isFirst() { return d_isFirst; }

  private:
    string               d_label;
    string               d_expr;
    AssertionSupportEnum d_level;
    bool                 d_isFirst;
};  /* class AssertionExpression */


/**
 * Contract comment/clause data.
 */
class ContractComment
{
  public:
    ContractComment(ContractCommentEnum t, PPIDirectiveType dt): 
      d_type(t), d_needsReturn(false), d_isInit(false), d_dirType(dt), 
      d_numExec(0) {}

    ~ContractComment() { d_aeList.clear(); }

    ContractCommentEnum type() { return d_type; }
    ContractClauseEnum clause() { return ContractCommentClause[d_type]; }
    void add(AssertionExpression ae) 
    { 
      d_aeList.push_front(ae); 
      if (ae.support() == AssertionSupport_EXECUTABLE) d_numExec += 1;
    }
    void setInit(bool init) { d_isInit = init; }
    bool isInit() { return d_isInit; }
    void setResult(bool needs) { d_needsReturn = needs; }
    bool needsResult() { return d_needsReturn; }
    list<AssertionExpression> getList() { return d_aeList; }
    void clear() { d_aeList.clear(); }
    int size() { return d_aeList.size(); }
    PPIDirectiveType directive() { return d_dirType; }
    int numExecutable() { return d_numExec; }

  private:
    ContractCommentEnum        d_type;
    list<AssertionExpression>  d_aeList;
    PPIDirectiveType           d_dirType;
    bool                       d_needsReturn;
    bool                       d_isInit;
    int                        d_numExec;
};  /* class ContractComment */


/*
 *************************************************************************
 * Utility Routines
 *************************************************************************
 */
class ContractsProcessor
{
  public:
    ContractsProcessor() : d_invariants(NULL) {}
    ~ContractsProcessor() { if (d_invariants != NULL) delete d_invariants; }

   int addPreChecks(SgFunctionDefinition* def, SgBasicBlock* body, 
     ContractComment* cc);

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
