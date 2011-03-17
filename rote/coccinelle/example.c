#include <stdio.h>
#include <stdlib.h>

void foo() {
  printf("foo\n");
  void *x;
  x = malloc(1);
  printf("ok\n");
  if(NULL == x) exit(1);
  free(x);
}

void bar() {
  printf("bar\n");
  void *y;
  y = malloc(2);
}

int main() {
  foo();
  bar();
  return 0;
}
