import sidl, ir, codegen
# Generator for testcases
# This is a one-shot script to generate a couple of testcases.
# It also doubles as a convoluted example of how to use BRAID to
# generate code in multiple languages.

# Measure two things:
# - Overhead of function call (struct passing)
# - Overhead of member access
#
# +--------------------------------------------------+ ^
# |                                                  | |
# |                                                  | |
# |                                                  | | time(Lang)-time(C)
# |                                                  |
# |                                                  |
# |                                                  |
# |                                                  |
# |                                                  |
# +--------------------------------------------------+
# 1 call                                           1000 calls
# 1000 accesses                                    1 access


def sidl_code(n, datatype):
    """
    Generate SIDL definition file

    SIDL code will look similar to this:         
    package s version 1.0  {                     
      struct Vector1 {                           
        float m1;                                
      }                                          
                                             
      class Benchmark {                          
        float run(in Vector1 a, out Vector1 b);  
      }                                          
    
    }                                            
    """
    return sidl.File([], [], [sidl.User_type([],
            sidl.Package("s", sidl.Version(1.0),
             [sidl.User_type([], sidl.Struct(sidl.Scoped_id([], 'Vector', ''),
                                  [sidl.Struct_item(sidl.Primitive_type(datatype),
                                                    ('m%d'%i))
                                   for i in range(1, n+1)])),
              sidl.User_type([], 
               sidl.Class(("Benchmark"), [], [], [],
                [sidl.Method(
                  sidl.Primitive_type(datatype), 
                  sidl.Method_name(("run"), ''), [],
                  [sidl.Arg([], sidl.in_, 
                            sidl.Scoped_id([], "Vector", ''), ("a")),
                   sidl.Arg([], sidl.inout,
                            sidl.Scoped_id([], "Vector", ''), ("b"))],
                  [], [], [], [],
                  'Run the benchmark')
                  ],
                   'Benchmark class'))],
                'Benchmark package')
        )])

#-----------------------------------------------------------------------
# benchmark kernels
#-----------------------------------------------------------------------
def dotproduct_expr(n, datatype):
    def add(a, b):
        return ir.Plus(a, b)

    def item(name):
        return ir.Struct_item(ir.Primitive_type(datatype), (name))

    a = ir.Struct(ir.Scoped_id([("s")], ('Vector'), ''),
         [ir.Struct_item(ir.Primitive_type(datatype), ('m%d'%i)) for i in range(1, n+1)], '')
    b = a
    e = reduce(add, map(lambda i:
                            ("*",
                             ir.Get_struct_item(a, (ir.deref, "a"), 'm%d'%i),
                             ir.Get_struct_item(b, (ir.deref, "b"), 'm%d'%i)),
                        range(1, n+1)))
    return ir.Stmt(ir.Return(e))

def const_access_expr(n, datatype):
    def add(a, b):
        return ir.Plus(a, b)

    def item(name):
        return ir.Struct_item(ir.Primitive_type(datatype), (name))

    t = ir.Struct(ir.Scoped_id([("s")], ('Vector'), ''), 
         [ir.Struct_item(ir.Primitive_type(datatype), ('m%d'%i)) for i in range(1, n+1)], '')
    e = reduce(add, map(lambda i:
                            ("*",
                             ir.Get_struct_item(t, (ir.deref, "a"), item('m%d'%(i%n+1))),
			     ir.Get_struct_item(t, (ir.deref, "b"), item('m%d'%(i%n+1)))),
                        range(1, 129)))
    return ir.Stmt(ir.Return(e))

def reverse_expr(n, datatype):
    'b_i = a_{n-i}'
    def item(name):
        return ir.Struct_item(ir.Primitive_type(datatype), (name))

    t = ir.Struct(ir.Scoped_id([("s")], ('Vector'), ''), 
         [ir.Struct_item(ir.Primitive_type(datatype), ('m%d'%i)) for i in range(1, n+1)], '')
    revs = [ir.Stmt(ir.Set_struct_item(t, (ir.deref, "b"), item('m%d'%i),
                       ir.Get_struct_item(t, (ir.deref, "a"), item('m%d'%(n-i+1)))))
            for i in range(1, n+1)]
    return revs+[ir.Stmt(ir.Return(retval(n, datatype)))]


def nop_expr(n, datatype):
    return ir.Stmt(ir.Return(retval(n, datatype)))

def bsort_expr(n, datatype):
    def assign(var, val):
        return ir.Stmt((ir.assignment, var, val))

    def item(name):
        return ir.Struct_item(ir.Primitive_type(datatype), (name))

    t = ir.Struct(ir.Scoped_id([("s")], ('Vector'), ''),
            [item('m%d'%i) for i in range(1, n+1)], '')
    a = ir.Arg([], ir.in_, t, 'a')
    b = ir.Arg([], ir.inout, t, 'b')
    copy = [ir.Stmt(ir.Set_struct_item(t, (ir.deref, "b"), item('m%d'%i),
                      ir.Get_struct_item(t, (ir.deref, "a"), item('m%d'%i))))
            for i in range(1, n+1)]
    sort = [(ir.var_decl, ir.pt_bool, ("swapped")),
            (ir.var_decl, ir.pt_int, ("tmp")),
            (ir.do_while, ("swapped"),
             # if A[i-1] > A[i]
             [assign(("swapped"), ir.Bool(ir.false))]+
             [[(ir.if_, 
                (ir.Infix_expr(ir.gt, 
                              ir.Get_struct_item(t, (ir.deref, "b"), item('m%d'%(i-1))),
                              ir.Get_struct_item(t, (ir.deref, "b"), item('m%d'%i)))),
                # swap( A[i-1], A[i] )
                # swapped = true
                [assign(("tmp"), 
                        ir.Get_struct_item(t, (ir.deref, "b"), item('m%d'%i))),
                 ir.Stmt(ir.Set_struct_item(t, (ir.deref, "b"), item('m%d'%i),
                           ir.Get_struct_item(t, (ir.deref, "b"), item('m%d'%(i-1))))),
                 ir.Stmt(ir.Set_struct_item(t, (ir.deref, "b"), item('m%d'%(i-1)),
                           ("tmp"))),
                 assign(("swapped"), ir.Bool(ir.true))])]
              for i in range(2, n+1)])]
    return copy+sort+[ir.Stmt(ir.Return(retval(n, datatype)))]

def retval(n, datatype):
    if datatype == "bool":     return (ir.Bool(ir.true))
    elif datatype == "int":    return (n)
    elif datatype == "float":  return (ir.Float(float(n)))
    elif datatype == "string": return (ir.Str(str(n)))
    else: raise

#-----------------------------------------------------------------------
# return a main.c for the client implementation
#-----------------------------------------------------------------------
def gen_main_c(n, datatype):
    t = codegen.CCodeGenerator().get_type(datatype)
    if datatype == "bool":
        init = '\n  '.join(["a.m%d = TRUE;"%i          	   for i in range(1, n+1)]
			  +["b.m%d = FALSE;"%i		   for i in range(1, n+1)])
    elif datatype == "float":				   
	init = '\n  '.join(["a.m%d = %f;"%(i, float(i))	   for i in range(1, n+1)]
			  +["b.m%d = %f;"%(i, float(i))	   for i in range(1, n+1)])
    elif datatype == "int":				   
	init = '\n  '.join(["a.m%d = %d;"%(i, float(n-i))  for i in range(1, n+1)]
			  +["b.m%d = %d;"%(i, float(n-i))  for i in range(1, n+1)])
    elif datatype == "string":
        init = '\n  '.join(['a.m%d = strdup("             %3d");'%(i, i) for i in range(1, n+1)]
			  +['b.m%d = strdup("             %3d");'%(i, i) for i in range(1, n+1)])
    else: raise
    return r"""
#include <stdlib.h>
#include <stdio.h>
#include "s_Benchmark.h"
#include "sidl_BaseInterface.h"
#include "sidl_Exception.h"
  
 int main(int argc, char** argv)
 {
   int i; long num_runs;
	  sidl_BaseInterface ex;
   s_Benchmark h = s_Benchmark__create(&ex); SIDL_CHECK(ex);
   struct s_Vector__data a, b;
   if (argc != 2) {
     fprintf(stderr,"Usage: %s <number of runs>\n", argv[0]);
     return 1;
   }
   
   num_runs = strtol(argv[1], (char **) NULL, 10);
   fprintf(stderr, "running %ld times\n", num_runs);

   /* Initialization */
   """+init+r"""
   /* Benchmarks */
   for (i=0; i<num_runs; ++i) {
     volatile """+t+r""" result = s_Benchmark_run(h, &a, &b, &ex); SIDL_CHECK(ex);
   }
   s_Benchmark_deleteRef(h, &ex); SIDL_CHECK(ex);
	  return 0;
EXIT: /* this is error handling code for any exceptions that were thrown */
  {
    char* msg;
    fprintf(stderr,"%s:%d: Error, exception caught\n",__FILE__,__LINE__);
    sidl_BaseInterface ignore = NULL;
    sidl_BaseException be = sidl_BaseException__cast(ex,&ignore);

    msg = sidl_BaseException_getNote(be, &ignore);
    fprintf(stderr,"%s\n",msg);
    sidl_String_free(msg);

    msg = sidl_BaseException_getTrace(be, &ignore);
    fprintf(stderr,"%s\n",msg);
    sidl_String_free(msg);

    sidl_BaseException_deleteRef(be, &ignore);
    SIDL_CLEAR(ex);
    return 1;
  }

}
"""


import subprocess, splicer, argparse, os, re
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
def main():
    cmdline = argparse.ArgumentParser(description='auto-generate struct benchmarks')
    cmdline.add_argument('i', metavar='n', type=int,
			 help='number of elements in the Vector struct')
    cmdline.add_argument('datatype', metavar='t', 
			 help='data type for the Vector struct')
    cmdline.add_argument('expr', metavar='expr', choices=['reverse', 'nop', 'bsort'],
			 help='benchmark expression to generate')
    # cmdline.add_argument('babel', metavar='babel',
    #			 help='the Babel executable')
    args = cmdline.parse_args()
    i = args.i
    datatype = args.datatype
    expr = args.expr
    if expr == 'reverse':
        benchmark_expr = reverse_expr
    elif expr == 'nop':
        benchmark_expr = nop_expr
    elif expr == 'bsort':
        benchmark_expr = bsort_expr
    else: raise

    print "-------------------------------------------------------------"
    print "generating servers"
    print "-------------------------------------------------------------"
    subprocess.check_call("mkdir -p out", shell=True)
    f = open('out/struct_%d_%s_%s.sidl'%(i,datatype,expr), "w")
    f.write(codegen.generate("SIDL", sidl_code(i, datatype)))
    f.close()
    languages = ["C", "CXX", "F77", "F90", "F03", "Java", "Python", "Chapel" ]
    for lang in languages:
        ext = {"C"      : "c", 
               "CXX"    : "cxx",
               "F77"    : "f",
               "F90"    : "F90",
               "F03"    : "F03",
               "Java"   : "java",
               "Python" : "py",
               "Chapel" : "chpl"}
        prefix = {"C"   : "s_", 
               "CXX"    : "s_",
               "F77"    : "s_",
               "F90"    : "s_",
               "F03"    : "s_",
               "Java"   : "s/",
               "Python" : "s/",
               "Chapel" : "s"}
        babel={"C"      : "babel", 
               "CXX"    : "babel",
               "F77"    : "babel",
               "F90"    : "babel",
               "F03"    : "babel",
               "Java"   : "babel",
               "Python" : "babel",
               "Chapel" : "braid"}


        print "generating", lang, i, datatype, expr, "..."

        cmd = """
          mkdir -p out/{lang}_{i}_{t}_{e} && cd out/{lang}_{i}_{t}_{e} &&
          {babel} -s{lang} --makefile ../struct_{i}_{t}_{e}.sidl
          """.format(lang=lang,i=i,babel=babel[lang],t=datatype,e=expr)
        #print cmd
        subprocess.check_call(cmd, shell=True)
        impl = ("out/{lang}_{i}_{t}_{e}/{prefix}{name}.{ext}".
                format(lang=lang, i=i, t=datatype, e=expr,
                       ext=ext[lang], prefix=prefix[lang], 
                       name='_Impl' if lang == 'Chapel' else 'Benchmark_Impl'))
        if lang == "Python":
            splicer_block = "run"
        else: splicer_block = "s.Benchmark.run"
        code = codegen.generate(lang, benchmark_expr(i,datatype))
        if code == None:
            raise Exception('Code generation failed')
        #print "splicing", impl
        #print code
        splicer.replace(impl, splicer_block, code) 

    print "-------------------------------------------------------------"
    print "generating client", i, datatype, expr, "..."
    print "-------------------------------------------------------------"
    cmd = """
      mkdir -p out/client_{i}_{t}_{e} && cd out/client_{i}_{t}_{e} &&
      {babel} -cC --makefile ../struct_{i}_{t}_{e}.sidl
      """.format(i=i,babel=babel["C"],t=datatype,e=expr)
    #print cmd
    subprocess.check_call(cmd, shell=True)
    f = open('out/client_%d_%s_%s/main.c'%(i,datatype,expr), "w")
    f.write(gen_main_c(i,datatype))
    f.close


    print "-------------------------------------------------------------"
    print "adapting client Makefile..."
    print "-------------------------------------------------------------"
    filename = 'out/client_%d_%s_%s/GNUmakefile'%(i,datatype,expr)
    os.rename(filename, filename+'~')
    dest = open(filename, 'w')
    src = open(filename+'~', 'r')

    for line in src:
        m = re.match(r'^(all *:.*)$', line)
        if m:
            dest.write(m.group(1)+
                       ' runC2C runC2CXX runC2F77 runC2F90 runC2F03 '
                       +'runC2Java runC2Python runC2Chapel\n')
            dest.write("CXX=`babel-config --query-var=CXX`\n"+
                       '\n'.join([
"""
runC2{lang}: lib$(LIBNAME).la ../{lang}_{i}_{t}_{e}/libimpl.la main.lo
\tbabel-libtool --mode=link $(CC) -static main.lo lib$(LIBNAME).la \
\t    ../{lang}_{i}_{t}_{e}/libimpl.la -o runC2{lang}
""".format(lang=lang, i=i, t=datatype, e=expr) for lang in languages[:6]+languages[7:]]))
            dest.write("""
runC2Python: lib$(LIBNAME).la ../Python_{i}_{t}_{e}/libimpl1.la main.lo
\tbabel-libtool --mode=link $(CC) -static main.lo lib$(LIBNAME).la \
\t    ../Python_{i}_{t}_{e}/libimpl1.la -o runC2Python
""".format(i=i,t=datatype,e=expr))
        else:
            dest.write(line)
    dest.close()
    src.close()

    print "-------------------------------------------------------------"
    print "generating benchmark script..."
    print "-------------------------------------------------------------"
    def numruns(t):
        if t == 'string': 
            return str(100001)
        return str(1000001)

    for lang in languages:
        f = open('out/client_%d_%s_%s/runC2%s.sh'%(i,datatype,expr, lang), 'w')
        f.write(r"""#!/usr/bin/bash
LIBDIR=`babel-config --query-var=libdir`
PYTHON_VERSION=`babel-config --query-var=PYTHON_VERSION`
PYTHONPATH_1=$LIBDIR/python$PYTHON_VERSION/site-packages:$PYTHONPATH
SIDL_VERSION=`babel-config --query-var=VERSION`
SIDL_DLL_PATH_1="$LIBDIR/libsidlstub_java.scl;$LIBDIR/libsidl.scl;$LIBDIR/libsidlx.scl"
export LD_LIBRARY_PATH="$LIBDIR:$LD_LIBRARY_PATH"

# echo "runC2{lang}({i})"

function count_insns {
   # measure the number of instructions of $1, save output to $2.all
   # perform one run with only one iteration and subtract that from the result 
   # to eliminate startup time
   perf stat -- $1 1 2>$2.base || (echo "FAIL1" >$2; echo FAIL1; exit 1)
   base=`grep instructions $2.base | awk '{print $1}'`
   perf stat -- $1 """+numruns(datatype)+r""" 2>$2.perf || (echo "FAILN" >$2; echo FAILN; exit 1)
   grep instructions $2.perf | awk "{print \$1-$base}" >> $2.all
}

function medtime {
   # measure the median running user time
   rm -f $2.all
   MAX=1 # 3 #### 10 see also 1:9 below!!!!!
   for I in `seq $MAX`; do
     echo "measuring $1 ($3@$4,$5) [$I/$MAX]"
     count_insns $1 $2
   done
   cat $2.all \
       | sort \
       | python -c 'import numpy,sys; \
           print numpy.mean( \
             sorted(map(lambda x: float(x), sys.stdin.readlines())) )#[1:9])' \
       >>$2
}
""")
        f.write('''
rm -f out{lang}
export SIDL_DLL_PATH="../{lang}_{i}_{t}_{e}/libimpl.scl;$SIDL_DLL_PATH_1"
export PYTHONPATH="../Python_{i}_{t}_{e}:$PYTHONPATH_1"
export CLASSPATH="../Java_{i}_{t}_{e}:$LIBDIR/sidl-$SIDL_VERSION.jar:$LIBDIR:$LIBDIR/sidlstub_$SIDL_VERSION.jar"
export BABEL_JVM_FLAGS=-XX:CompileThreshold=1
medtime ./runC2{lang} out{lang} {i} {t} {e}
'''.format(lang=lang,i=i,t=datatype,e=expr))
        f.close()
    # end for langs

    f = open('out/client_%d_%s_%s/combine.sh'%(i,datatype,expr), 'w')
    f.write("#!/bin/sh") 
    f.write("echo %d "%i+' '.join(['`cat out%s`'%lang 
                                   for lang in languages])+' >times\n')
    f.close()


if __name__ == '__main__':
    try:
        main()
    except:
        # Invoke the post-mortem debugger
        import pdb, sys
        print sys.exc_info()
        pdb.post_mortem()
