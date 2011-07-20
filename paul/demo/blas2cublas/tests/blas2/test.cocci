//Begin Header insertion patch.
//This patch is taken from coccinelle.
//demos file first.cocci


@@ 
identifier a,b,c,d,e;

@@ 

<...
- cblas_sdot(a,b,c,d,e) 
+ 42
...>

