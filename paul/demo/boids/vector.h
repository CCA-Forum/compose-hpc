#ifndef VECTOR_H
#define VECTOR_H

typedef struct {
  double x;
  double y;
} Vec;

double vec_length(Vec a);
Vec vec_scale(Vec a, double scale);
Vec vec_normalize(Vec a);
Vec vec_add(Vec a, Vec b);
Vec vec_sub(Vec a, Vec b);
Vec vec_limit(Vec a, double limit);
Vec vec_rand();

#endif
