/* 
 * File:          Knapsack.cpp
 * Description:   The Knapsack class.
 */

#include <iostream>
#include <string.h>
#include "Knapsack.hpp"

using namespace std;

static const char* L_MAX_WEIGHTS = "Cannot exceed maximum number of weights.";
static const char* L_POS_WEIGHTS = "Non-positive weights are NOT supported.";

/*
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 ToDo/TBD:  Do the contracts belong here or in the header?
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 */

/* %CONTRACT INVARIANT all_pos_weights: onlyPosWeights(); */

/**
 *  Default constructor
 */
void 
Knapsack() {
  d_nextIndex = 0;
  memset(d_weights, 0, (size_t)(MAX_WEIGHTS*sizeof(int)));
  return;
}

/**
 * Initialize the knapsack with the specified weights, w.
 */
/* CONTRACT REQUIRE 
    pos_weights: (weights != null) implies all(weights > 0, len); 
 */
/* CONTRACT ENSURE 
    has_new_weights: hasWeights(weights, len); 
 */
void
Knapsack::initialize(unsigned int* weights, unsigned int len)
{
  unsigned int i;
  bool         onlyPos;

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

/**
 * Return TRUE if all weights in the knapsack are positive;
 * otherwise, return FALSE.
 */
/* CONTRACT ENSURE 
    side_effect_free: is pure;
 */
bool
Knapsack::onlyPosWeights() {
  return onlyPos(d_weights, d_nextIndex);
}

/**
 * Return TRUE if all of the specified weights, w, are in the knapsack
 * or there are no specified weights; otherwise, return FALSE.
 */
/* CONTRACT REQUIRE 
    pos_weights: (weights != null) implies all(weights > 0, len); 
 */
/* CONTRACT ENSURE 
    side_effect_free: is pure;
 */
bool
Knapsack::hasWeights(unsigned int* weights, unsigned int len) {
  return sameWeights(d_weights, d_nextIndex, weights, len);
}


/**
 * Return TRUE if there is a solution for the specified target
 * weight; otherwise, return FALSE.  Recall a solution is a
 * subset of weights that total exactly to the specified target
 * weight.
 */
/* CONTRACT ENSURE 
    side_effect_free: is pure;
 */
bool
Knapsack::hasSolution(unsigned int t) {
  return solve(d_weights, t, 0, d_nextIndex);
}


/*
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 ToDo/TBD: Should contracts be placed on private methods?  Probably
 not if SIDL is going to be performing the translation; however, in
 that case the contracts should be in the header not here...
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 */

/*
 * Determine if all entries in the list are positive, returning TRUE
 * if they are or FALSE if they are not.
 */
/* CONTRACT REQUIRE 
    pos_weights: (weights != null) implies all(weights > 0, len); 
 */
/* CONTRACT ENSURE 
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

/*
 * Check to see if the two lists match where order does not matter.
 */
/* CONTRACT REQUIRE 
    pos_weights: (nW != null) implies all(nW > 0, lenW); 
    pos_weights: (nS != null) implies all(nS > 0, lenS); 
 */
/* CONTRACT ENSURE 
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


/*
 * Recursive implementation of the simplified knapsack problem.
 *
 * Based on the algorithm defined in "Data Structures and
 * Algorithms" by Aho, Hopcroft, and Ullman (c) 1983.
 */
/* CONTRACT REQUIRE 
    pos_weights: (weights != null) implies all(weights > 0, n); 
 */
/* CONTRACT ENSURE 
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
    has = true;
  } else {
    has = solve(weights, t, i+1, n);
  }
  return has;
} /* solve */
