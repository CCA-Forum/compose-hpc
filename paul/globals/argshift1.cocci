//
// Adding arguments systematically to all the functions in a library (for the build on a given system) would introduce a semantic diversification that disables or converts to crashes any externally built binary (typically injected via shell code) that expects to find a 'standard' build associated with a given library name or symbol
// startup python
@initialize:python@
body_idlist=[]
decl_idlist=[]
stage2 = open("stage2.cocci",'w+',1)

// match function definition and extend args
@anyfuncdef@
type T;
identifier id;
position p0;
@@
T id(@p0 ... ) {...}
// do a bit of python with definition
@ script:python @
id << anyfuncdef.id;
T << anyfuncdef.T;
p0 << anyfuncdef.p0;
@@
body_idlist.append( (str(id), T) )
print "id=", id
print "location", p0[0].file, p0[0].line, p0[0].column
print ""
// do a bit of rewrite with definition
@@
identifier anyfuncdef.id;
fresh identifier argadded;
@@

id(
+  struct dummy_insert *argadded,  
...) {
+ if (argadded != 4096) { exit(2); }
+ {
...
+ }
}


// match function decl
// note T must appear in the patch text or it is nearly ambiguous with a call to id()
@anyfuncdecl@
type T;
identifier id;
parameter list[n] args;
@@
T id( args ) ;

// rewrite with decl
@@
identifier anyfuncdecl.id;
type anyfuncdecl.T;
parameter list[n] anyfuncdecl.args;
@@

T id(
+  struct dummy_insert *argadded,
 args ) ;


// python w/decl
@ script:python @
id << anyfuncdecl.id;
T << anyfuncdecl.T;
@@
body_idlist.append( (str(id), T) )
print "id=", id
print ""

// shutdown python
@finalize:python@
print >> stage2, "defs", body_idlist
print >> stage2, "decls", decl_idlist
stage2.close()
