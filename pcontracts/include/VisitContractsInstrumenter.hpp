/**
 * \internal
 * File:          VisitContractsInstrumenter.hpp
 * Author:        T. Dahlgren
 * Created:       2012 November 1
 * Last Modified: 2013 October 28
 * \endinternal
 *
 * @file
 * @brief
 * Second contract enforcement instrumentation example.
 *
 * @details
 * This example translates structured contract comments associated with any
 * AST node into contract checks, contract-related runtime calls, or comments,
 * depending on the annotation and associated AST node.  Preconditions
 * and postconditions must still be associated with function nodes; however,
 * all other contract annotations can be associated with other node types.
 *
 * Only rudimentary (string) check of the associated expression is performed to
 * assess whether the expression is (likely to be) a basic C expression or a 
 * more advanced expression using special operators, keywords, or functions.
 *
 * @htmlinclude copyright.html
 */

#ifndef include_Visit_Contracts_Instrumenter_hpp
#define include_Visit_Contracts_Instrumenter_hpp

#include "rose.h"
#include "ContractsProcessor.hpp"


class VisitContractsInstrumenter : public AstSimpleProcessing
{
  public:
    /**
     * Constructor.
     */
    VisitContractsInstrumenter(Sg_File_Info* fileInfo); 

    /**
     * Destructor.
     */
    virtual ~VisitContractsInstrumenter();

    /**
     * Process the current AST node passed by the front end, identifying
     * and printing individual contract clause assertions.
     *
     * @param[in,out]  node  Current AST node.
     */
    void virtual visit(SgNode* node);

    /**
     * Output the number of contract-related statements added.
     */
    virtual void atTraversalEnd(void);


  private:
    /** The contracts processor. */
    ContractsProcessor*  d_processor;

    /**
     * Information on the file currently being processed.  This is
     * useful for eliminating processing of some front end AST nodes
     * (e.g., ROSE's numeric_traits.h).
     */
    Sg_File_Info* d_fileInfo;

    /** The last line processed. */
    long  d_lastLine;

    /** Global processing status. */
    bool  d_globalDone;

    /** The number of contract-related statements added. */
    int  d_num;

    VisitContractsInstrumenter& operator=( 
       const VisitContractsInstrumenter& vci);

    VisitContractsInstrumenter(VisitContractsInstrumenter& vci);

}; /* VisitContractsInstrumenter */

#endif /* include_Visit_Contracts_Instrumenter_hpp */
