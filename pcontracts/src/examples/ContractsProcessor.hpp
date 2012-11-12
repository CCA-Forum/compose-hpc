/**
 * File:           ContractsProcessor.hpp
 * Author:         T. Dahlgren
 * Created:        2012 November 1
 * Last Modified:  2012 November 12
 *
 *
 * @file
 * @section DESCRIPTION
 * Basic contract clause processing utilities.
 *
 *
 * @section SOURCE
 * This code was originally part of the initial ContractInstrumenter.cpp,
 * which was renamed to RoutineContractInstrumenter.cpp.
 *
 * 
 * @section COPYRIGHT
 * Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Tamara Dahlgren <dahlgren1@llnl.gov>.
 * 
 * LLNL-CODE-473891.
 * All rights reserved.
 * 
 * This software is part of COMPOSE-HPC. See http://compose-hpc.sourceforge.net/
 * for details.  Please read the COPYRIGHT file for Our Notice and for the 
 * BSD License.
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
