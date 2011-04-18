@def@
identifier s,x,y,z;
type T1,T2,T3;
@@
struct s {
- T1 x;
- T2 y;
- T3 z;
+ T1 *x;
+ T2 *y;
+ T3 *z;
};


@decl@
identifier def.s,k;
@@
- struct s *k;
+ struct s k;


@@
function foo;
identifier def.s,k,x;
expression E1;
@@
foo(...,
- struct s *k
+ struct s k
,...) {
<...
- k[E1].x
+ k.x[E1]
...>
}


@@
identifier decl.k,def.s,def.x,def.y,def.z;
expression E;
type def.T1,def.T2,def.T3;
@@
- k = malloc(E * sizeof(struct s));
+ k.x = malloc(E * sizeof(T1));
+ k.y = malloc(E * sizeof(T2));
+ k.z = malloc(E * sizeof(T3));
...
- free(k);
+ free(k.x);
+ free(k.y);
+ free(k.z);


//@@
//identifier x,y;
//expression E1,E2;
//@@
//for(...;...;...) {
//<...
//- x[E1].y = E2;
//+ x.y[E1] = E2;
//...>
//}
