int foo() {
  int *x;
  double *y;
  unsigned char c;
  short s;
  long l;
  unsigned long long ll;
  x = malloc(10);
  // if (x == null) exit(1);   <--- insert this
  return 0;
}
