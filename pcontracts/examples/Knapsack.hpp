/**
 * File:          Knapsack.hpp
 * Description:   The Knapsack class.
 */

#ifndef included_Knapsack_hpp
#define included_Knapsack_hpp

/*
 * Maximum number of weights.
 */
static const unsigned int MAX_WEIGHTS=10;

#define M_THROW(TP, MSG, MNM) {\
TP _ex = TP::_create(); \
_ex.setNote(MSG); \
_ex.add(__FILE__, __LINE__, MNM); \
throw _ex; \
}

/*
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 ToDo/TBD:  Do the contracts belong here or in the source?
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 */

class Knapsack 
{
  protected:
    unsigned int d_nextIndex;
    unsigned int d_weights[MAX_WEIGHTS];
  
  public:
    /**
     *  Default constructor
     */
    Knapsack();
  
    /**
     *  Destructor
     */
    virtual ~Knapsack() { }
  
  public:
    /**
     * Initialize the knapsack with the specified weights, weights.
     *
     * @param weights  The weights of the items that could be placed in
     *                   the knapsack.
     * @param len      The length, or number, of weights in the list.
     */
    void
    initialize(unsigned int* weights, unsigned int len);
  
    /**
     * Return TRUE if all weights in the knapsack are positive;
     * otherwise, return FALSE.
     */
    bool
    onlyPosWeights();
  
    /**
     * Return TRUE if all of the specified weights, w, are in the knapsack
     * or there are no specified weights; otherwise, return FALSE.
     *
     * @param weights  The weights of the items that could be placed in
     *                   the knapsack.
     * @param len      The length, or number, of weights in the list.
     */
    bool
    hasWeights(unsigned int* weights, unsigned int len);
  
    /**
     * Return TRUE if there is a solution for the specified target
     * weight; otherwise, return FALSE.  Recall a solution is a
     * subset of weights that total exactly to the specified target
     * weight.
     *
     * @param t  The target weight of items to be carried in the knapsack.
     */
    bool
    hasSolution(unsigned int t);
};  /* end class Knapsack */

#endif /* included_Knapsack_hpp */
