/**
 * \internal
 * File:  WeightsList.hpp
 * \endinternal
 *
 * @file
 * @brief
 * Class header, with labeled contracts, for maintaining a list of weights.
 *
 * @details
 * This class is intended as an example triggering the invariant violation when
 * contracts are enforced.
 *
 * @htmlinclude copyright.html
 */

#ifndef included_Weights_List_hpp
#define included_Weights_List_hpp

namespace Examples {

/**
 * Maximum number of weights.
 */
static const unsigned int MAX_WEIGHTS=10;

class WeightsList
{
  protected:
    /**
     * The fixed size list of the weights of available items.
     */
    unsigned int d_weights[MAX_WEIGHTS];
  
    /**
     * The next index in the fixed size list of possible weights.
     */
    unsigned int d_nextIndex;

  public:
    /**
     *  Returns an instance of the WeightsList class.
     */
    WeightsList();
  
    /**
     *  Deletes the instance.
     */
    virtual ~WeightsList();
  
  public:
    /**
     * Initialize the knapsack with the specified available weights.
     *
     * @param[in] weights  The weights of available items.
     * @param[in] len      The length, or number, of weights in the list.
     */
    void
    initialize(const unsigned int* weights, unsigned int len);
  
    /**
     * Determine whether all weights of available items are positive.
     *
     * @return  Returns true if they are all non-zero; otherwise, returns false.
     */
    bool
    onlyPosWeights();

    /**
     * Print any weights in the list.
     */
    void
    printWeights();

};  /* end class WeightsList */

}; /* end namespace Examples */

#endif /* included_Weights_List_hpp */
