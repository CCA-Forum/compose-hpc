#include <stdio.h>

/*
 %CONTRACT REQUIRE 
    has_weights: weights != 0;
    has_target: t > 0;
    has_n: n > 0;
 */
/*
 %CONTRACT ENSURE
    valid_result: _inrange(result, 0, 1);
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
}
