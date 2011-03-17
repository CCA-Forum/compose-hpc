@@
identifier x;
expression E;
constant int c;
@@
x = malloc(E);
+ if(x == NULL) exit(1);
... when != if(x == NULL) exit(c);
