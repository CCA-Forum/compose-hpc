int foo() {
  int *x;
  int *y;
  x = malloc(10);
  y = malloc(12);
  // if (x == null) exit(1);   <--- insert this
  return 0;
}
