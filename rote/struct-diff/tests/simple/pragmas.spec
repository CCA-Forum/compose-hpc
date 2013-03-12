void foo() {
  int x,y,z,a;

#pragma rulegen METAVARS(y,z)
- x = a*(y+z);
+ x = (a*y)+(a*z);
}
