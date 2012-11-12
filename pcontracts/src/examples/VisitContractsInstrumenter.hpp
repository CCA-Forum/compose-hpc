/**
 * File:          VisitContractsInstrumenter.hpp
 * Author:        T. Dahlgren
 * Created:       2012 November 1
 * Last Modified: 2012 November 12
 *
 *
 * @section DESCRIPTION
 * Experimental PAUL contracts instrumenter (via ROSE visitor pattern).
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

#ifndef include_Visit_Contracts_Instrumenter_hpp
#define include_Visit_Contracts_Instrumenter_hpp

#include "rose.h"


class VisitContractsInstrumenter : public AstSimpleProcessing
{
  public:
    /**
     * Constructor.
     */
    VisitContractsInstrumenter() { d_processor = ContractsProcessor(); };

    /**
     * Add requisite include file(s).
     *
     * @param project  The Sage project representing the initial AST of the
     *                   file(s).
     *
     * @return         The processing statue: 0 for success, non-0 for failure.
     */
    /* TLD/TODO:  Change to be in atTraversalStart() */
     int addIncludes(SgProject* project);

    /**
     * Process the current AST node passed by the front end, identifying
     * and printing individual contract clause assertions.
     *
     * @param node  Current AST node.
     */
    void virtual visit(SgNode* node);

    /* TLD/TODO:  Add atTraversalEnd() */

  private:
    /** The contracts processor. */
    ContractsProcessor  d_processor;

}; /* VisitContractsInstrumenter */

#endif /* include_Visit_Contracts_Instrumenter_hpp */
