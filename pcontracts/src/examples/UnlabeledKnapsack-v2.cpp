/**
 * \internal
 * File:  UnlabeledKnapsack-v2.cpp
 * \endinternal
 *
 * @file
 * @brief
 * Class implementation, with unlabeled contracts, for printing a solution to 
 * the knapsack problem.
 *
 * @details
 * Class used for printing a solution to the knapsack problem for any 
 * given target based on a known set of possible weights, where the
 * size of the list is restricted.
 *
 * Contract annotations in this version of the program do NOT contain optional
 * labels.
 *
 * @htmlinclude knapsackSource.html
 * @htmlinclude copyright.html
 */

#include <iostream>
#include <string.h>
#include <stdlib.h>
#include "UnlabeledKnapsack-v2.hpp"

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


/* %CONTRACT INVARIANT onlyPosWeights(); */


Examples::UnlabeledKnapsack::UnlabeledKnapsack() {
  d_nextIndex = 0;
  memset(d_weights, 0, (size_t)(MAX_WEIGHTS*sizeof(int)));
  return;
} /* UnlabeledKnapsack */


/* %CONTRACT REQUIRE 
    is initialization;
    ((weights!=NULL) and (len>0)) implies pce_all(weights>0, len); 
 */
/* %CONTRACT ENSURE hasWeights(weights, len); */
void
Examples::UnlabeledKnapsack::initialize(unsigned int* weights, unsigned int len)
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
        cerr << "\nERROR: " << L_POS_WEIGHTS << "\n";
      }
    } else {
      // ToDo/TBD:  Throw an exception (L_MAX_WEIGHTS)
      cerr << "\nERROR: " << L_MAX_WEIGHTS << "\n";
    }
  }

  return;  
} /* initialize */


/* %CONTRACT ENSURE is pure; */
bool
Examples::UnlabeledKnapsack::onlyPosWeights() {
  return onlyPos(d_weights, d_nextIndex);
} /* onlyPosWeights */


/* %CONTRACT REQUIRE 
  ((weights!=NULL) and (len>0)) implies pce_all(weights>0, len);
 */
/* %CONTRACT ENSURE is pure; */
bool
Examples::UnlabeledKnapsack::hasWeights(unsigned int* weights, 
                                        unsigned int len) 
{
  return sameWeights(d_weights, d_nextIndex, weights, len);
} /* hasWeights */


/* %CONTRACT REQUIRE t > 0; */
/* %CONTRACT ENSURE is pure; */
bool
Examples::UnlabeledKnapsack::hasSolution(unsigned int t) {
  return solve(d_weights, t, 0, d_nextIndex);
} /* hasSolution */


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
    ((weights!=NULL) and (len>0)) implies pce_all(weights>0, len);
 */
/* %CONTRACT ENSURE is pure; */
bool
onlyPos(unsigned int* weights, unsigned int len) 
{
  unsigned int i;
  bool         isPos = false;

  /*
   * Routine _should_ be directly protecting itself from bad inputs rather
   * than relying on assertions whose enforcement can be disabled (and will
   * only result in executable checks with the Visitor version of the 
   * instrumentor); however, needed some plausible excuse for using the
   * assertion 'contract'...
   */

  /* %CONTRACT ASSERT
      weights!=NULL;
      len>0;
   */

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
    ((nW!=NULL) and (lenW>0)) implies pce_all(nW>0, lenW); 
    ((nS!=NULL) and (lenS>0)) implies pce_all(nS>0, lenS); 
 */
/* %CONTRACT ENSURE is pure; */
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
    pos_target: t > 0;
    ((weights!=NULL) and (n>0)) implies pce_all(weights>0, n); 
 */
/* %CONTRACT ENSURE is pure; */
bool
solve(unsigned int* weights, unsigned int t, unsigned int i, unsigned int n) {
  bool has = false;

  /*
   * Routine _should_ be directly protecting itself from bad inputs rather
   * than relying on assertions whose enforcement can be disabled (and will
   * only result in executable checks with the Visitor version of the 
   * instrumentor); however, needed some plausible excuse for using the
   * assertion 'contract'...
   */

  /* %CONTRACT ASSERT
      weights!=NULL;
      n>0;
   */

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
 * Perform a single solve, relying on the UnlabeledKnapsack class to output 
 * the result from a successful run.
 *
 * @param ksack  The knapsack instance.
 * @param t      The target weight.
 */
/* %CONTRACT REQUIRE ksack != NULL; */
void
runIt(Examples::UnlabeledKnapsack* ksack, unsigned int t)
{
  cout << "Solution for target=" << t <<"?: ";
  if ( (ksack != NULL) && !ksack->hasSolution(t) ) {
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
/* %CONTRACT INIT */
/* %CONTRACT FINAL */
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

  Examples::UnlabeledKnapsack* ksack = new Examples::UnlabeledKnapsack();
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
