/**
 * \internal
 * File:  WeightsList.cpp
 * \endinternal
 *
 * @file
 * @brief
 * Class implementation, with labeled contracts, for maintaining a list of weights.
 *
 * @details
 * This class is intended as an example triggering the invariant violation when
 * contracts are being enforced.
 *
 * @htmlinclude copyright.html
 */

#include <iostream>
#include <string.h>
#include <stdlib.h>
#include "WeightsList.hpp"

using namespace std;

static const char* L_MAX_WEIGHTS = "Cannot exceed maximum number of weights.";


bool
onlyPos(const unsigned int* weights, unsigned int len);


/* %CONTRACT INVARIANT all_pos_weights: onlyPosWeights(); */


Examples::WeightsList::WeightsList() : d_nextIndex(0) {
  memset(d_weights, 0, (size_t)(MAX_WEIGHTS*sizeof(int)));
  return;
}


Examples::WeightsList::~WeightsList() {}


/* %CONTRACT REQUIRE 
    initialization: is initialization;
 */
void
Examples::WeightsList::initialize(
  /* in */ const unsigned int* weights,
  /* in */ unsigned int        len)
{
  if (weights != NULL) {
    if (len <= MAX_WEIGHTS) {
      for (unsigned int i=0; i<len; i++) {
        d_weights[i] = weights[i];
      }
      d_nextIndex = len;
    } else {
      // ToDo/TBD:  Throw an exception (L_MAX_WEIGHTS)
      cerr << "\nERROR: " << L_MAX_WEIGHTS << "\n";
    }
  }

  return;  
}

/* %CONTRACT ENSURE 
    side_effect_free: is pure;
 */
bool
Examples::WeightsList::onlyPosWeights() {
  return onlyPos(d_weights, d_nextIndex);
}


/**
 * Print any weights in the list.
 */
void
Examples::WeightsList::printWeights()
{
  cout << "WeightsList contains: ";
  for (int i=0; i<d_nextIndex-1; i++) {
    cout << d_weights[i] << ", ";
  }
  if (d_nextIndex>1) {
    cout << "and " << d_weights[d_nextIndex-1] << ".\n";
  }

  return;
} /* printWeights */


/**
 * Determine whether the weights of all of the available items are
 * positive.
 *
 * @param[in] weights  The weights of the items that could be added to the 
 *                       knapsack.
 * @param[in] len      The length, or number, of weights in the list.
 * @return             Returns true if all entries are non-zero or there are
 *                       no weights; otherwise, returns false.
 */
/* %CONTRACT ENSURE 
    side_effect_free: is pure;
 */
bool
onlyPos(
  /* in */ const unsigned int* weights,
  /* in */ unsigned int        len) 
{
  bool isPos = true;

  if (weights != NULL) {
    for (unsigned int i=0; (i<len) && isPos; i++) {
      if (weights[i] <= 0) {
        isPos = false;
      }
    }
  }

  return isPos;
} /* onlyPos */


/**
 * Test Driver, which accepts an optional target value resulting in a single
 * solve.  If no target value is provided, then multiple solutions will be
 * generated for targets in a predetermined range.
 */
/* %CONTRACT INIT */
/* %CONTRACT FINAL */
int 
main(int argc, char **argv) {
  if (argc>1) {
    cout << "USAGE: " << argv[0] << "\n";
    exit(1);
  }

  /* Check should fail only if contracts enforced. */
  Examples::WeightsList* wlist = new Examples::WeightsList();
  if (wlist != NULL) {
    unsigned int num=5;
    unsigned int weights[5] = { 1, 8, 0, 4, 3 };

    wlist->initialize(weights, num);
    wlist->printWeights();

    delete wlist;
  }

  return 0;
} /* main */
