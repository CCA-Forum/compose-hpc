/**
 * File:          Knapsack.hpp
 *
 *
 * @file
 * @section DESCRIPTION
 * Class used for printing a solution to the knapsack problem for any
 * given target based on a known set of possible weights, where the 
 * size of the list is restricted.
 *
 * The implementation uses a recursive algorithm based on that defined 
 * in "Data Structures and Algorithms" by Aho, Hopcroft, and Ullman (c)
 * 1983.
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

#ifndef included_Knapsack_hpp
#define included_Knapsack_hpp

namespace Examples {

/**
 * Maximum number of weights.
 */
static const unsigned int MAX_WEIGHTS=10;

class Knapsack 
{
  protected:
    /**
     * The next index in the fixed size list of possible weights.
     */
    unsigned int d_nextIndex;

    /**
     * The fixed size list of the weights of available items.
     */
    unsigned int d_weights[MAX_WEIGHTS];
  
  public:
    /**
     *  Returns an instance of the Knapsack class.
     */
    Knapsack();
  
    /**
     *  Deletes the instance.
     */
    virtual ~Knapsack() { }
  
  public:
    /**
     * Initialize the knapsack with the specified available weights.
     *
     * @param weights  The weights of available items.
     * @param len      The length, or number, of weights in the list.
     */
    void
    initialize(unsigned int* weights, unsigned int len);
  
    /**
     * Determine whether all weights of available items are positive.
     *
     * @return  Returns true if they are all non-zero; otherwise, returns false.
     */
    bool
    onlyPosWeights();

    /**
     * Determine whether the specified weights match those currently
     * available for the knapsack.
     *
     * @param weights  The weights of available items.
     * @param len      The length, or number, of weights in the list.
     * @return         Returns true if the specified weights match
     *                   those currently available; otherwise, returns
     *                   false.
     */
    bool
    hasWeights(unsigned int* weights, unsigned int len);
  
    /**
     * Determine whether a solution exists such that a subset of weights
     * totals exactly the specified target weight.
     *
     * @param  t  The target weight of items to be carried in the knapsack.
     * @return    Returns true if there is a solution for the specified 
     *              target; otherwise, returns false.
     */
    bool
    hasSolution(unsigned int t);
};  /* end class Knapsack */

}; /* end namespace Examples */

#endif /* included_Knapsack_hpp */
