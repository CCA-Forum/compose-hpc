/**
 * \internal
 * File:  WeightsList2.hpp
 * \endinternal
 *
 * @file
 * @brief
 * Class header, with labeled contracts, for maintaining a list of weights.
 *
 * @details
 * This class is intended as an example that has contracts defined in the 
 * header, which cannot be automatically instrumented as enforcement checks
 * until Rose includes header information in the AST.
 *
 * @htmlinclude copyright.html
 */

#ifndef included_Weights_List_2_hpp
#define included_Weights_List_2_hpp

namespace Examples {

/**
 * Maximum number of weights.
 */
static const unsigned int MAX_WEIGHTS=10;

/* %CONTRACT INVARIANT all_pos_weights: onlyPosWeights(); */


class WeightsList2
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
     *  Returns an instance of the WeightsList2 class.
     */
    WeightsList2();
  
    /**
     *  Deletes the instance.
     */
    virtual ~WeightsList2();
  
  public:
    /**
     * Initialize the knapsack with the specified available weights.
     *
     * @param[in] weights  The weights of available items.
     * @param[in] len      The length, or number, of weights in the list.
     */
    /* %CONTRACT REQUIRE 
        initialization: is initialization;
     */
    void
    initialize(const unsigned int* weights, unsigned int len);
  
    /**
     * Determine whether all weights of available items are positive.
     *
     * @return  Returns true if they are all non-zero; otherwise, returns false.
     */
    /* %CONTRACT ENSURE 
        side_effect_free: is pure;
     */
    bool
    onlyPosWeights();

    /**
     * Print any weights in the list.
     */
    void
    printWeights();

};  /* end class WeightsList2 */

}; /* end namespace Examples */

#endif /* included_Weights_List_2_hpp */
