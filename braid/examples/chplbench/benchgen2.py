import sidl, ir, codegen, chapel
# Generator for testcases
# This is a one-shot script to generate a couple of testcases.
# It also doubles as a convoluted example of how to use BRAID to
# generate code in multiple languages.
# cf. ../structbench/benchgen.py

def sidl_code(n, datatype):
    return """
    package s version 1.0 {{
      class Benchmark {{
        void run({args});
      }}

    }}
    """.format(args=', '.join([
    'in {type} a{n}, out {type} b{n}'.format(type=datatype, n=i)
    for i in range(0, n)]))

def sidl_code_sum(n, datatype):
    return """
    package s version 1.0 {{
      class Benchmark {{
        {type} run({args});
      }}

    }}
    """.format(type=datatype, args=', '.join([
    'in {type} a{n}'.format(type=datatype, n=i)
    for i in range(0, n)]))


#-----------------------------------------------------------------------
# benchmark kernels
#-----------------------------------------------------------------------

def copy_expr(n, datatype):
    def assign(var, val):
        return ir.Stmt(ir.Set_arg(var, val))

    # unhold = []
    # if lang == "Java":
    #     unhold = [ir.Stmt("{t} b{n}_held = b{n}.get()".format(t=datatype, n=i))
    #               for i in range(0, n)]

    return [assign('b%d'%i, 'a%d'%i) for i in range(0, n)]


def sum_expr(n, datatype):
    e = reduce(ir.Plus, ['a%d'%i for i in range(0, n)], 0)
    return [ir.Stmt(ir.Return(e))]


#-----------------------------------------------------------------------
# return a main.chpl for the client implementation
#-----------------------------------------------------------------------
def gen_main_chpl(n, datatype, bench):
    t = chapel.ChapelCodeGenerator.type_map[datatype]
    if datatype == "bool":
        init = '\n  '.join(["var a%d: bool = true;"%i              for i in range(0, n)]
                          +["var b%d: bool = false;"%i             for i in range(0, n)])
    elif datatype == "float":
        init = '\n  '.join(["var a%d: real(32) = %f : real(32);"%(i, float(i))   for i in range(0, n)]
                          +["var b%d: real(32) = %f : real(32);"%(i, float(i))   for i in range(0, n)])
    elif datatype == "int":
        init = '\n  '.join(["var a%d: int(32) = %d;"%(i, float(n-i))  for i in range(0, n)]
                          +["var b%d: int(32) = %d;"%(i, float(n-i))  for i in range(0, n)])
    elif datatype == "string":
        init = '\n  '.join(['var a%d = "                            %3d";'%(i, i) for i in range(0, n)]
                          +['var b%d = "                            %3d";'%(i, i) for i in range(0, n)])
    else: raise Exception("data type")

    if bench == 'sum':
        args = ', '.join(['a{n}'.format(n=i) for i in range(0,n)])
    else:
        args = ', '.join(['a{n}, b{n}'.format(n=i) for i in range(0,n)])

    return r"""
use s;
use sidl;
config var num_runs:int(32) = 1;
writeln("running "+num_runs+" times");

var ex: sidl.BaseException;
var server = new s.Benchmark(ex);

/* Benchmarks */
[i in 0..num_runs] {
  /* Initialization */
  """+init+r"""
  server.run(%s, ex);
}
"""%args
    


import subprocess, splicer, argparse, os, re
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
def main():
    cmdline = argparse.ArgumentParser(description='auto-generate chpl benchmarks')
    cmdline.add_argument('i', metavar='n', type=int,
                         help='number of elements in the Vector chpl')
    cmdline.add_argument('datatype', metavar='t',
                         help='data type for the Vector chpl')
    cmdline.add_argument('expr', metavar='expr', choices=['copy', 'sum'],
                         help='benchmark expression to generate')
    # cmdline.add_argument('babel', metavar='babel',
    #                    help='the Babel executable')
    args = cmdline.parse_args()
    i = args.i
    datatype = args.datatype
    babel = 'babel' #args.babel
    expr = args.expr
    if expr == 'copy':
        benchmark_expr = copy_expr
    if expr == 'sum':
        benchmark_expr = sum_expr
        sidl_code = sidl_code_sum
    else: raise

    print "-------------------------------------------------------------"
    print "generating servers"
    print "-------------------------------------------------------------"
    subprocess.check_call("mkdir -p out", shell=True)
    f = open('out/chpl_%d_%s_%s.sidl'%(i,datatype,expr), "w")
    f.write(codegen.generate("SIDL", sidl_code(i, datatype)))
    f.close()
    languages = ["C", "CXX", "F77", "F90", "F03", "Java", "Python"]
    for lang in languages:
        ext = {"C"      : "c",
               "CXX"    : "cxx",
               "F77"    : "f",
               "F90"    : "F90",
               "F03"    : "F03",
               "Java"   : "java",
               "Python" : "py"}
        prefix = {"C"   : "s_",
               "CXX"    : "s_",
               "F77"    : "s_",
               "F90"    : "s_",
               "F03"    : "s_",
               "Java"   : "s/",
               "Python" : "s/"}

        print "generating", lang, i, datatype, expr, "..."
        cmd = """
          mkdir -p out/{lang}_{i}_{t}_{e} && cd out/{lang}_{i}_{t}_{e} &&
          {babel} -s{lang} --makefile ../chpl_{i}_{t}_{e}.sidl
          """.format(lang=lang,i=i,babel=babel,t=datatype,e=expr)
        #print cmd
        subprocess.check_call(cmd, shell=True)
        impl = ("out/{lang}_{i}_{t}_{e}/{prefix}Benchmark_Impl.{ext}".
                format(lang=lang, i=i, t=datatype, e=expr,
                       ext=ext[lang], prefix=prefix[lang]))
        if lang == "Python":
            splicer_block = "run"
        else: splicer_block = "s.Benchmark.run"
        code = codegen.generate(lang, benchmark_expr(i,datatype))

        if code == None:
            raise Exception('Code generation failed')
        print "splicing", impl
        splicer.replace(impl, splicer_block, code)

    print "-------------------------------------------------------------"
    print "generating client", i, datatype, expr, "..."
    print "-------------------------------------------------------------"
    cmd = """
      mkdir -p out/client_{i}_{t}_{e} && cd out/client_{i}_{t}_{e} &&
      {babel} -cChapel --makefile ../chpl_{i}_{t}_{e}.sidl
      """.format(i=i,babel='braid',t=datatype,e=expr)
    #print cmd
    subprocess.check_call(cmd, shell=True)
    f = open('out/client_%d_%s_%s/main.chpl'%(i,datatype,expr), "w")
    f.write(gen_main_chpl(i,datatype,expr))
    f.close

    print "-------------------------------------------------------------"
    print "generating benchmark script..."
    print "-------------------------------------------------------------"
    def numruns(t):
        if t == 'string':
            return str(100001)
        return    str(1000001)

    f = open('out/client_%d_%s_%s/runAll.sh'%(i,datatype,expr), 'w')
    f.write(r"""#!/usr/bin/bash
PYTHONPATH_1="`/usr/bin/python -c "from distutils.sysconfig import get_python_lib; print get_python_lib(prefix='/home/prantl1/work/babel/install',plat_specific=1) + ':' + get_python_lib(prefix='/home/prantl1/work/babel/install')"`:$PYTHONPATH"
LIBDIR=`babel-config --query-var=libdir`
PYTHON_VERSION=`babel-config --query-var=PYTHON_VERSION`
SIDL_VERSION=`babel-config --query-var=VERSION`
SIDL_DLL_PATH_1="$LIBDIR/libsidlstub_java.scl;$LIBDIR/libsidl.scl;$LIBDIR/libsidlx.scl"
export LD_LIBRARY_PATH="$LIBDIR:$LD_LIBRARY_PATH"

echo "runAll($i)"

function count_insns {
   # measure the number of instructions of $1, save output to $2.all
   # perform one run with only one iteration and subtract that from the result
   # to eliminate startup time
   perf stat -- $1 --num_runs=1 2>$2.base || (echo "FAIL" >$2; exit 1)
   base=`grep instructions $2.base | awk '{print $1}'`
   perf stat -- $1 --num_runs="""+numruns(datatype)+r""" 2>$2.perf || (echo "FAIL" >$2; exit 1)
   grep instructions $2.perf | awk "{print \$1-$base}" >> $2.all
}

function medtime {
   # measure the median running user time
   rm -f $2.all
   MAX=10
   for I in `seq $MAX`; do
     echo "measuring $1 ($3@$4,$5) [$I/$MAX]"
     # echo SIDL_DLL_PATH=$SIDL_DLL_PATH
     # echo PYTHONPATH=$PYTHONPATH
     # /usr/bin/time -f %U -a -o $2.all $1 || (echo "FAIL" >$2; exit 1)
     count_insns $1 $2
   done
   cat $2.all \
       | sort \
       | python -c 'import numpy,sys; \
           print numpy.median( \
             map(lambda x: float(x), sys.stdin.readlines()))' \
       >>$2
}
""")
    for lang in languages:
        f.write('''
rm -f out{lang}
export SIDL_DLL_PATH="../{lang}_{i}_{t}_{e}/libimpl.scl;$SIDL_DLL_PATH_1"
export PYTHONPATH="../Python_{i}_{t}_{e}:$PYTHONPATH_1"
export CLASSPATH="../Java_{i}_{t}_{e}:$LIBDIR/sidl-$SIDL_VERSION.jar:$LIBDIR/sidlstub_$SIDL_VERSION.jar"
medtime ./runChapel2{lang} out{lang} {i} {t} {e}
'''.format(lang=lang,i=i,t=datatype,e=expr))

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
