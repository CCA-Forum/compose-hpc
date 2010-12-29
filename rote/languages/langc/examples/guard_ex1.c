int foo() {
  int *x;
  x = malloc(10);
  // if (x == null) exit(1);   <--- insert this
  return 0;
}
