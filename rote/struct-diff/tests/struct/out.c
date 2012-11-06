typedef struct foo {
  int *a;
  double *b;
} foo_t;

foo_t foos;

void allocate(int n) {
  foos.a = malloc(sizeof(int)*n);
  foos.b = malloc(sizeof(double)*n);
}


