/**
 * \internal
 * File:  knapsack.c
 * \endinternal
 *
 * @file
 * @brief
 * C program, with labeled contracts, for printing a solution to the knapsack 
 * problem.
 *
 * @details
 * A program for printing a solution to the knapsack problem for any
 * given target based on a known set of possible weights, where the 
 * size of the list is restricted.
 *
 * Contract annotations in this version of the program contain optional labels.
 *
 * @htmlinclude knapsackSource.html
 * @htmlinclude copyright.html
 */

#include <stdio.h>
#include <stdlib.h>


/**
 * \privatesection
 */
/**
 * Determine if there is a solution to the knapsack problem based on
 * the list of weights for avaialbe items, target weight, and current 
 * position.
 *
 * @param[in] weights  The weights of available items.
 * @param[in] t        The target weight.
 * @param[in] i        The current weight entry.
 * @param[in] n        The number of items (or weights) in the list.
 * @return             Returns 1 if a solution is detected; otherwise, returns 
 *                       0.
 */
/*
 %CONTRACT REQUIRE 
    has_weights: weights != 0;
    has_n: n > 0;
 */
/*
 %CONTRACT ENSURE
    valid_result: pce_inrange(pce_result, 0, 1);
 */
int 
knapsack(
  /* in */ unsigned int* weights, 
  /* in */ unsigned int  t, 
  /* in */ unsigned int  i, 
  /* in */ unsigned int  n)
{
  int has = 0;

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
    has = 1;
  } else if (i >= n) {
    has = 0;
  } else if (knapsack(weights, t-weights[i], i+1, n)) {
    printf("%d ", weights[i]);
    has = 1;
  } else {
    has = knapsack(weights, t, i+1, n);
  }

  return has;
} /* knapsack */


/**
 * \publicsection
 */

/**
 * Perform a single solve, relying on the Knapsack class to output the
 * result from a successful run.
 *
 * @param[in] weights  The weights of available items.
 * @param[in] t        The target weight.
 * @param[in] num      The number of items (or weights) in the knapsack.
 */
/*
 %CONTRACT REQUIRE 
    has_weights: weights != 0;
    has_length: num > 0;
 */
void
runIt(
  /* in */ unsigned int* weights, 
  /* in */ unsigned int  t, 
  /* in */ unsigned int  num)
{
  printf("\nSolution for target=%d?: ", t);
  if (!knapsack(weights, t, 0, num)) {
    printf("None");
  } else if (t == 0) {
    printf("N/A");
  }
  printf("\n");

  return;
} /* runIt */


/**
 * Test driver, which accepts an optional target value resulting in a single
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
    printf("Assuming targets ranging from %d to %d.\n\n", min, max);
    t = -1;
  } else if (argc==2) {
    t = atoi(argv[1]);
    if (t < 0) {
      printf("Replacing the negative target entered (%d) with 0.\n\n", t);
      t = 0;
    }
  } else {
    printf("USAGE: %s [<target-value]\n", argv[0]);
    exit(1);
  }

  printf("Knapsack contains: ");
  for (i=0; i<num-1; i++) {
    printf("%d, ", weights[i]);
  }
  printf("and %d.\n", weights[num-1]);

  if (t != -1) {
    runIt(weights, t, num);
  } else {
    for (i=min; i<=max; i++) {
      runIt(weights, i, num);
    }
  }

  return 0;
} /* main */
