/**
 * File:  knapsack.c
 *
 * @file
 * @section DESCRIPTION
 * A program for printing a solution to the knapsack problem for any
 * given target based on a known set of possible weights, where the 
 * size of the list is restricted.
 *
 * A recursive algorithm is implemented based on that defined in "Data
 * Structures and Algorithms" by Aho, Hopcroft, and Ulman (c) 1983.
 *
 * @section LICENSE
 * TBD
 */

#include <stdio.h>
#include <stdlib.h>


/**
 * Determine if there is a solution to the knapsack problem based on
 * the list of weights for avaialbe items, target weight, and current 
 * position.
 *
 * @param weights  The weights of available items.
 * @param t        The target weight.
 * @param i        The current weight entry.
 * @param n        The number of items (or weights) in the list.
 * @return         Returns 1 if a solution is detected; otherwise, returns 0.
 */
/*
 %CONTRACT REQUIRE 
    has_weights: weights != 0;
    has_n: n > 0;
 */
/*
 %CONTRACT ENSURE
    valid_result: _inrange(_result, 0, 1);
 */
int 
knapsack(unsigned int* weights, unsigned int t, unsigned int i, unsigned int n)
{
  int has = 0;

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
 * Perform a single solve, relying on the Knapsack class to output the
 * result from a successful run.
 *
 * @param weights  The weights of available items.
 * @param t        The target weight.
 * @param num      The number of items (or weights) in the knapsack.
 */
/*
 %CONTRACT REQUIRE 
    has_weights: weights != 0;
    has_length: num > 0;
 */
void
runIt(unsigned int* weights, unsigned int t, unsigned int num)
{
  printf("Solution for target=%d?: ", t);
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
  printf("and %d.\n\n", weights[num-1]);

  if (t != -1) {
    runIt(weights, t, num);
  } else {
    for (i=0; i<=max; i++) {
      runIt(weights, i, num);
    }
  }

  return 0;
} /* main */
