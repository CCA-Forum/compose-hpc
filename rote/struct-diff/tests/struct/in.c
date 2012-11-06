typedef struct foo {
  int a;
  double b;
} foo_t;

foo_t *foos;

void allocate(int n) {
  foos = malloc(sizeof(foo_t)*n);
}


