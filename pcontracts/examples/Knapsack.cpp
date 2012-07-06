/* 
 * File:  Knapsack.cpp
 *
 * @file
 * @section DESCRIPTION
 * Class used for printing a solution to the knapsack problem for any 
 * given target based on a known set of possible weights, where the
 * size of the list is restricted.
 *
 * @section LICENSE
 * TBD
 */

#include <iostream>
#include <string.h>
#include <stdlib.h>
#include "Knapsack.hpp"

using namespace std;

static const char* L_MAX_WEIGHTS = "Cannot exceed maximum number of weights.";
static const char* L_POS_WEIGHTS = "Non-positive weights are NOT supported.";


bool
onlyPos(unsigned int* weights, unsigned int len);

bool
sameWeights(unsigned int* nW, unsigned int lenW, 
            unsigned int* nS, unsigned int lenS);

bool
solve(unsigned int* weights, unsigned int t, unsigned int i, unsigned int n);


/*
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 ToDo/TBD:  Do the contracts belong here or in the header?

 Header would be better for documentation purposes; however,
 the implementation makes instrumentation easier.
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 */

/* %CONTRACT INVARIANT all_pos_weights: onlyPosWeights(); */


Examples::Knapsack::Knapsack() {
  d_nextIndex = 0;
  memset(d_weights, 0, (size_t)(MAX_WEIGHTS*sizeof(int)));
  return;
}

/* %CONTRACT REQUIRE 
    pos_weights: (weights != null) implies all(weights > 0, len); 
 */
/* %CONTRACT ENSURE 
    has_new_weights: hasWeights(weights, len); 
 */
void
Examples::Knapsack::initialize(unsigned int* weights, unsigned int len)
{
  unsigned int i;

  if (weights != NULL) {
    if (len <= MAX_WEIGHTS) {
      if (onlyPos(weights, len)) {
        for (i=0; i<len; i++) {
          d_weights[i] = weights[i];
        }
        d_nextIndex = len;
      } else {
        // ToDo/TBD:  Throw a "BadWeight" exception (L_POS_WEIGHTS)
        std::cerr << "ERROR: " << L_POS_WEIGHTS << std::endl;
      }
    } else {
      // ToDo/TBD:  Throw an exception (L_MAX_WEIGHTS)
      std::cerr << "ERROR: " << L_MAX_WEIGHTS << std::endl;
    }
  }

  return;  
}

/* %CONTRACT ENSURE 
    side_effect_free: is pure;
 */
bool
Examples::Knapsack::onlyPosWeights() {
  return onlyPos(d_weights, d_nextIndex);
}

/* %CONTRACT REQUIRE 
    pos_weights: (weights != null) implies all(weights > 0, len); 
 */
/* %CONTRACT ENSURE 
    side_effect_free: is pure;
 */
bool
Examples::Knapsack::hasWeights(unsigned int* weights, unsigned int len) {
  return sameWeights(d_weights, d_nextIndex, weights, len);
}


/* %CONTRACT ENSURE 
    side_effect_free: is pure;
 */
bool
Examples::Knapsack::hasSolution(unsigned int t) {
  return solve(d_weights, t, 0, d_nextIndex);
}


/*
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 ToDo/TBD: Should contracts be placed on private methods?  Probably
 not if SIDL is going to be performing the translation; however, in
 that case the contracts should be in the header not here...
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 */

/**
 * Determine whether the weights of all of the available items are
 * positive.
 *
 * @param weights  The weights of the items that could be added to the 
 *                   knapsack.
 * @param len      The length, or number, of weights in the list.
 * @return         Returns true if they are all non-zero; otherwise,
 *                   returns false.
 */
/* %CONTRACT REQUIRE 
    pos_weights: (weights != null) implies all(weights > 0, len); 
 */
/* %CONTRACT ENSURE 
    side_effect_free: is pure;
 */
bool
onlyPos(unsigned int* weights, unsigned int len) 
{
  unsigned int i;
  bool         isPos = false;

  if (len > 0) {
    isPos = true;
    for (i=0; (i<len) && isPos; i++) {
      if (weights[i] <= 0) {
        isPos = false;
      }
    }
  }

  return isPos;
} /* onlyPos */

/**
 * Determine whether the weights in the two lists match, where order
 * does not matter.
 *
 * @param nW    The weights of the items that could be added to the knapsack.
 * @param lenW  The length, or number, of weights in nW.
 * @param nS    The weights of the items that could be added to the knapsack.
 * @param lenS  The length, or number, of weights in nS.
 * @return      Returns true if the values in the two lists match; 
 *                otherwise, returns false.
 */
/* %CONTRACT REQUIRE 
    pos_weights: (nW != null) implies all(nW > 0, lenW); 
    pos_weights: (nS != null) implies all(nS > 0, lenS); 
 */
/* %CONTRACT ENSURE 
    side_effect_free: is pure;
 */
bool
sameWeights(unsigned int* nW, unsigned int lenW, 
            unsigned int* nS, unsigned int lenS)
{
  bool     same = false;
  unsigned int* p;
  unsigned int  i, j, w;

  if ((nW != NULL) && (nS != NULL)) {
    if (lenW == lenS && lenW > 0) {
      p = (unsigned int*)malloc(lenW*sizeof(unsigned int));
      if (p) {
        memset(p, 0, (size_t)(lenW*sizeof(unsigned int)));
        for (i=0; i<lenW; i++) {
          w = nS[i];
          for (j=0; j<lenW; j++) {
            if ((w == nW[j]) && !p[j]) {
              p[j] = 1;
              break;
            }
          }
        }
        same = onlyPos(p, lenW);
        free(p);
      }
    }  /* else weights list size mismatch so assume will false */
  }  /* else no input weights provided so automatically false */

  return same;
} /* sameWeights */


/**
 * Determine if there is a solution to the problem for the target weight.  
 * This is a recursive implementation of the simplified knapsack problem
 * based on the algorithm defined in "Data Structures and Algorithms" by 
 * Aho, Hopcroft, and Ullman (c) 1983.
 *
 * @param weights  The weights of the items that could be added to the 
 *                   knapsack.
 * @param t        The desired, or target, weight of items to carry in
 *                   the knapsack.
 * @param i        The current entry in the list.
 * @param n        The number of weights in the list.
 * @return         Returns true if the solution has been found based on
 *                   the specified entry; otherwise, returns false.
 */
/* %CONTRACT REQUIRE 
    pos_weights: (weights != null) implies all(weights > 0, n); 
 */
/* %CONTRACT ENSURE 
    side_effect_free: is pure;
 */
bool
solve(unsigned int* weights, unsigned int t, unsigned int i, unsigned int n) {
  bool has = false;

  if (t==0) {
    has = true;
  } else if (i >= n) {
    has = false;
  } else if (solve(weights, t-weights[i], i+1, n)) {
    cout << weights[i] << " ";
    has = true;
  } else {
    has = solve(weights, t, i+1, n);
  }

  return has;
} /* solve */


/**
 * Perform a single solve, relying on the Knapsack class to output the
 * result from a successful run.
 *
 * @param ksack  The knapsack instance.
 * @param t      The target weight.
 */
void
runIt(Examples::Knapsack* ksack, unsigned int t)
{
  cout << "Solution for target=" << t <<"?: ";
  if (!ksack->hasSolution(t)) {
    cout << "None";
  } else if (t == 0) {
    cout << "N/A";
  }
  cout << "\n";

  return;
} /* runIt */


/**
 * Test Driver, which accepts an optional target value resulting in a single
 * solve.  If no target value is provided, then multiple solutions will be
 * generated for targets in a predetermined range.
 */
int 
main(int argc, char **argv) {
  int t;
  unsigned int i, num=7, min=0, max=20;
  unsigned int weights[7] = { 1, 8, 6, 5, 20, 4, 15 };

  if (argc==1) {
    cout << "Assuming targets ranging from " << min << " to " << max << ".\n\n";
    t = -1;
  } else if (argc==2) {
    t = atoi(argv[1]);
    if (t < 0) {
      cout << "Replacing the negative target entered (" << t << ") with 0.\n\n";
      t = 0;
    } 
  } else {
    cout << "USAGE: " << argv[0] << " [<target-value>]\n";
    exit(1);
  }

  Examples::Knapsack* ksack = new Examples::Knapsack();
  if (ksack != NULL) {
    ksack->initialize(weights, num);

    cout << "Knapsack contains: ";
    for (i=0; i<num-1; i++) {
      cout << weights[i] << ", ";
    }
    cout << "and " << weights[num-1] << ".\n\n";

    if (t != -1) {
      runIt(ksack, t);
    } else {
      for (i=min; i<=max; i++) {
        runIt(ksack, i);
      }
    }

    delete ksack;
  }

  return 0;
} /* main */
