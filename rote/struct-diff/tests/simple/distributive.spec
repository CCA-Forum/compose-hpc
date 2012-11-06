int foo(int a, int b, int c) {
  int x;

-  x = (a+b)*c;
+  x = a*c + b*c;
  return x;
}
