#include <stdlib.h>
#include <stdio.h>

struct s {
  double a;
  double b;
};

int foo(struct s *x) {
  printf("%lf\n",x[i].b);
  x[i].a = 0;
  return 5;
}

int main() {
  int n = 10;
  struct s *x;
  void *z;
  x = malloc(n * sizeof(struct s));
  z = malloc(10 * sizeof(struct s));
  int i;
  for(i=0; i < n; i++) {
    x[i].a = i;
    x[i].b = i*2;
  }
  return 0;
}
