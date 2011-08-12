//
// Adding arguments systematically to all the functions in a library (for the build on a given system) would introduce a semantic diversification that disables or converts to crashes any externally built binary (typically injected via shell code) that expects to find a 'standard' build associated with a given library name or symbol. This script changes the signatures and generates data needed to generate a script needed to fix the call sites.
// startup python
@initialize:python@
import pickle
# for debugging
body_idlist=[]
# for data collection
funcset= set()
# ignore uninitialized pickle case
try:
  pkl_file = open('data.pkl', 'rb')
  tmpset=pickle.load(pkl_file)
  pkl_file.close()
  funcset = tmpset
except:
  pass

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
print "id=", str(id)
print "location", p0[0].file, p0[0].line, p0[0].column
print ""

// do a bit of rewrite with definition
@@
identifier anyfuncdef.id;
@@

id(
+  struct dummy_insert *argadded,  
...) {
+ if (argadded != (struct dummy_insert*)4096) { exit(2); }
...
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

// shutdown python, generate callsite repair script
@finalize:python@
for (id,t) in body_idlist:
  funcset.add(id)
output=open('data.pkl','wb')
pickle.dump(funcset, output)
output.close()

// stage2 = open("stage2.cocci","a+",1)
// for (id,t) in body_idlist:
// print "ID%s" % (id)
// print >> stage2, """
// @fix_%s@
// expression P;
// @@
// -%s(P)
// +%s( ((struct dummy_insert *)4096), P)
// 
// """ % (id, id,id)
// stage2.close()

