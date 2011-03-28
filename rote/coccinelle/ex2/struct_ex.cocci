@def@
identifier s,x,y;
type T1,T2;
@@
struct s {
- T1 x;
- T2 y;
+ T1 *x;
+ T2 *y;
};


@decl@
identifier def.s,z;
@@
- struct s *z;
+ struct s z;


@@
function foo;
identifier def.s,x,y;
expression E1,E2;
@@
foo(
- struct s *x
+ struct s x
 ) {
<...
- x[E1].y
+ x.y[E1]
...>
}


@@
identifier decl.z,def.x,def.y,def.s;
expression E;
type def.T1,def.T2;
@@
- z = malloc(E * sizeof(struct s));
+ z.x = malloc(E * sizeof(T1));
+ z.y = malloc(E * sizeof(T2));


@@
identifier x,y;
expression E1,E2;
@@
for(...;...;...) {
<...
- x[E1].y = E2;
+ x.y[E1] = E2;
...>
}
