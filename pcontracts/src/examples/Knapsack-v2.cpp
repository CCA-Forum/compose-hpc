/**
 * \internal
 * File:  Knapsack-v2.cpp
 * \endinternal
 *
 * @file
 * @brief
 * Class implementation, with labeled contracts, for printing a solution to
 * the knapsack problem.
 *
 * @details
 * Class used for printing a solution to the knapsack problem for any 
 * given target based on a known set of possible weights, where the
 * size of the list is restricted.
 *
 * Contract annotations in this version of the program contain optional labels.
 *
 * @htmlinclude knapsackSource.html
 * @htmlinclude copyright.html
 */

#include <iostream>
#include <string.h>
#include <stdlib.h>
#include "Knapsack-v2.hpp"

using namespace std;

static const char* L_MAX_WEIGHTS = "Cannot exceed maximum number of weights.";
static const char* L_POS_WEIGHTS = "Non-positive weights are NOT supported.";


bool
onlyPos(const unsigned int* weights, unsigned int len);

bool
sameWeights(const unsigned int* nW, unsigned int lenW, 
            const unsigned int* nS, unsigned int lenS);

bool
solve(unsigned int* weights, unsigned int t, unsigned int i, unsigned int n);


/* %CONTRACT INVARIANT all_pos_weights: Examples::Knapsack::onlyPosWeights(); */


Examples::Knapsack::Knapsack() : d_nextIndex(0) {
  memset(d_weights, 0, (size_t)(MAX_WEIGHTS*sizeof(int)));
  return;
}


Examples::Knapsack::~Knapsack() {}

/*
 * @pre ((weights!=NULL) and (len>0)) implies onlyPos(weights, len)
 */
/* %CONTRACT REQUIRE 
    pos_weights: 
        !((weights!=NULL) && (len>0)) || onlyPos(weights,len);
    initialization: is initialization;
 */
/* %CONTRACT ENSURE has_new_weights: hasWeights(weights, len); */
void
Examples::Knapsack::initialize(
  /* in */ const unsigned int* weights,
  /* in */ unsigned int        len)
{
  /*
   * Routine _should_ be directly protecting itself from bad inputs rather
   * than relying on assertions whose enforcement can be disabled (and will
   * only result in executable checks with the Visitor version of the 
   * instrumentor); however, needed some plausible excuse for using the
   * assertion 'contract'...
   */

  /* %CONTRACT ASSERT
      given_weights: weights!=NULL;
      one_or_more_weights: len>0;
   */

  if (weights != NULL) {
    if (len <= MAX_WEIGHTS) {
      if (onlyPos(weights, len)) {
        for (unsigned int i=0; i<len; i++) {
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
}

/* %CONTRACT ENSURE side_effect_free: is pure; */
bool
Examples::Knapsack::onlyPosWeights() {
  return onlyPos(d_weights, d_nextIndex);
}

/*
 * @pre ((weights!=NULL) and (len>0)) implies onlyPos(weights, len)
 */

/* %CONTRACT REQUIRE 
  pos_weights: !((weights!=NULL) && (len>0)) || onlyPos(weights, len);
 */
/* %CONTRACT ENSURE side_effect_free: is pure; */
bool
Examples::Knapsack::hasWeights(
  /* in */ const unsigned int* weights, 
  /* in */ unsigned int        len) 
{
  return sameWeights(d_weights, d_nextIndex, weights, len);
}


/* %CONTRACT REQUIRE pos_target: t > 0; */
bool
Examples::Knapsack::hasSolution(unsigned int t) {
  return solve(d_weights, t, 0, d_nextIndex);
}


/**
 * Print any weights in the knapsack.
 */
void
Examples::Knapsack::printWeights()
{
  cout << "Knapsack contains: ";
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
 * @return             Returns true if they are all non-zero; otherwise,
 *                       returns false.
 *
 * NOTE: The following inappropriate precondition, if generated, would lead 
 * to an infinite loop since it calls itself.
 *
 * @pre ((weights!=NULL) and (n>0)) implies onlyPos(weights, n)
 */
/* %CONTRACT REQUIRE 
    pos_weights: !((weights!=NULL) && (n>0)) || onlyPos(weights, n); 
 */
/* %CONTRACT ENSURE side_effect_free: is pure; */
bool
onlyPos(
  /* in */ const unsigned int* weights,
  /* in */ unsigned int        len) 
{
  bool         isPos = false;

  /*
   * Routine _should_ be directly protecting itself from bad inputs rather
   * than relying on assertions whose enforcement can be disabled (and will
   * only result in executable checks with the Visitor version of the 
   * instrumentor); however, needed some plausible excuse for using the
   * assertion 'contract'...
   */

  /* %CONTRACT ASSERT
      given_weights: weights!=NULL;
      one_or_more_weights: len>0;
   */

  if (len > 0) {
    isPos = true;
    for (unsigned int i=0; (i<len) && isPos; i++) {
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
 * @param[in] nW    The weights of the items that could be added to the 
 *                    knapsack.
 * @param[in] lenW  The length, or number, of weights in nW.
 * @param[in] nS    The weights of the items that could be added to the 
 *                    knapsack.
 * @param[in] lenS  The length, or number, of weights in nS.
 *
 * @return          Returns true if the values in the two lists match; 
 *                    otherwise, returns false.
 *
 * @pre ((nW!=NULL) and (lenW>0)) implies onlyPos(nW, lenW)
 * @pre ((nS!=NULL) and (lenS>0)) implies onlyPos(nS, lenS)
 */
/* %CONTRACT REQUIRE 
    pos_w_weights: !((nW!=NULL) && (lenW>0)) || onlyPos(nW, lenW); 
    pos_s_weights: !((nS!=NULL) && (lenS>0)) || onlyPos(nS, lenS); 
 */
/* %CONTRACT ENSURE side_effect_free: is pure; */
bool
sameWeights(
  /* in */ const unsigned int* nW, 
  /* in */ unsigned int        lenW, 
  /* in */ const unsigned int* nS, 
  /* in */ unsigned int        lenS)
{
  bool same = false;

  if ((nW != NULL) && (nS != NULL)) {
    if (lenW == lenS && lenW > 0) {
      unsigned int* p = (unsigned int*)malloc(lenW*sizeof(unsigned int));
      if (p) {
        memset(p, 0, (size_t)(lenW*sizeof(unsigned int)));
        for (unsigned int i=0; i<lenW; i++) {
          unsigned int w = nS[i];
          for (unsigned int j=0; j<lenW; j++) {
            if ((w == nW[j]) && !p[j]) {
              p[j] = 1;
              break;
            }
          }
        }
        same = onlyPos(p, lenW);
        free(p);
      }
    }  /* else weights list size mismatch so assume will be false */
  }  /* else no input weights provided so automatically false */

  return same;
} /* sameWeights */


/**
 * Determine if there is a solution to the problem for the target weight.  
 * This is a recursive implementation of the simplified knapsack problem
 * based on the algorithm defined in "Data Structures and Algorithms" by 
 * Aho, Hopcroft, and Ullman (c) 1983.
 *
 * @param[in] weights  The weights of the items that could be added to the 
 *                       knapsack.
 * @param[in] t        The desired, or target, weight of items to carry in
 *                       the knapsack.
 * @param[in] i        The current entry in the list.
 * @param[in] n        The number of weights in the list.
 * @return             Returns true if the solution has been found based on
 *                       the specified entry; otherwise, returns false.
 *
 * @pre ((weights!=NULL) and (n>0)) implies onlyPos(weights, n)
 */
/* %CONTRACT REQUIRE 
    pos_weights: !((weights!=NULL) && (n>0)) || onlyPos(weights, n); 
 */
/* %CONTRACT ENSURE side_effect_free: is pure; */
bool
solve(
  /* in */ unsigned int* weights, 
  /* in */ unsigned int  t, 
  /* in */ unsigned int  i, 
  /* in */ unsigned int  n) 
{
  bool has = false;

  /*
   * Routine _should_ be directly protecting itself from bad inputs rather
   * than relying on assertions whose enforcement can be disabled (and will
   * only result in executable checks with the Visitor version of the 
   * instrumentor); however, needed some plausible excuse for using the
   * assertion 'contract'...
   */

  /* %CONTRACT ASSERT
      given_weights: weights!=NULL;
      one_or_more_weights: n>0;
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
 * Perform a single solve, relying on the Knapsack class to output the
 * result from a successful run.
 *
 * @param[in] ksack  The knapsack instance.
 * @param[in] t      The target weight.
 */
/* %CONTRACT REQUIRE has_sack: ksack != NULL; */
void
runIt(
  /* in */ Examples::Knapsack* ksack, 
  /* in */ unsigned int        t)
{
  cout << "\nSolution for target=" << t <<"?: ";
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
  unsigned int min=0, max=20;

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
    unsigned int num=7;
    unsigned int weights[7] = { 1, 8, 6, 5, 20, 4, 15 };

    ksack->initialize(weights, num);
    ksack->printWeights();

    if (t != -1) {
      runIt(ksack, t);
    } else {
      for (int i=min; i<=max; i++) {
        runIt(ksack, i);
      }
    }

    delete ksack;
  }

  return 0;
} /* main */
