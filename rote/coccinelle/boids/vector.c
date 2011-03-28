/*****************************************************************************
  Vector stuff
*****************************************************************************/

#include <stdlib.h>
#include <math.h>
#include "vector.h"

double vec_length(Vec a) {
  return sqrt(a.x * a.x + a.y * a.y);
}

Vec vec_scale(Vec a, double scale) {
  Vec b;
  b.x = a.x * scale;
  b.y = a.y * scale;
  return b;
}

Vec vec_normalize(Vec a) {
  return vec_scale(a,1.0/vec_length(a));
}

Vec vec_add(Vec a, Vec b) {
  Vec c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

Vec vec_sub(Vec a, Vec b) {
  Vec c;
  c.x = a.x - b.x;
  c.y = a.y - b.y;
  return c;
}

Vec vec_limit(Vec a, double limit) {
  if(vec_length(a) > limit) {
    return vec_scale(vec_normalize(a),limit);
  } else {
    return a;
  }
}

// Biased but it doesn't matter here
Vec vec_rand() {
  Vec a;
  a.x = (double)(rand() - RAND_MAX / 2);
  a.y = (double)(rand() - RAND_MAX / 2);
  return vec_normalize(a);
}
