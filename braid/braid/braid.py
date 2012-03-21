#!/usr/bin/env python
# -*- python -*-
## @package braid
#
# Command line handling for the BRAID tool
#
# Please report bugs to <adrian@llnl.gov>.
#
# \authors <pre>
#
# Copyright (c) 2011, 2012 Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Written by Adrian Prantl <adrian@llnl.gov>.
#  
# LLNL-CODE-473891.
# All rights reserved.
#  
# This file is part of BRAID. For details, see 
# http://compose-hpc.sourceforge.net/. 
# Please read the COPYRIGHT file for Our Notice and
# for the BSD License.
#
# </pre>
#
# \mainpage
# <h1>Welcome to Braid!</h1>
# (a.k.a. Babel 3)
#
# This is BRAID, the <b>B</b>RAID system for <b>r</b>ewriting
# <b>a</b>bstract <b>i</b>ntermediate <b>d</b>escriptions, a new tool
# that has a command line interface similar to that of Babel. BRAID is
# implemented in Python, making it very portable itself.  BRAID is a
# multi-faceted, term-based system for generating language
# interoperability glue code designed and developed as part of the
# COMPOSE-HPC project (http://compose-hpc.sourceforge.net/) to be a
# reusable component of software composability tools.
#
# From a user's perspective, BRAID is the tool that generates glue
# code for parallel PGAS languages, while Babel handles traditional
# HPC languages. Eventually, we intend to make this distinction
# invisible to the end user by launching both through the same front
# end.  Our Chapel language interoperability tool is the first of
# several applications envisioned for BRAID including optimized
# language bindings for a reduced set of languages and building
# multi-language interfaces to code without SIDL interface
# descriptions.
#
# The most important difference between BRAID and Babel is how the
# language backends are designed: In Babel each code generator is a
# fixed-function Java class that builds all the glue code out of
# strings. BRAID, on the other hand, creates glue code in a high-level
# <em>language-independent</em> intermediate representation (IR). This
# intermediate representation is then passed to a code generator which
# translates it into actual high-level code. At the moment there are
# code generators for C and Chapel, and also initial versions for
# Fortran, Java and Python. This architecture offers a higher
# flexibility than the static approach of Babel: For example,
# (object-)method calls in Babel need to be resolved by looking up the
# address of the method in a virtual function pointer table. Since
# Chapel has no means of dealing with function pointers (it implements
# its own object system instead), BRAID's Chapel code generator will
# generate a piece of C code to do the virtual function call <em>on
# the fly</em>, and place a static call to this helper function in
# lieu of the virtual function call.
# 
# Using this system we can reduce the number of times the language
# barrier is crossed to the minimum, leading to more code generated in
# the higher-level language, which again enables the compiler to do a
# better job at optimizing the program.
#
# Similar to Babel, BRAID can also be instructed to generate a
# <em>Makefile</em> that is used to compile both program and glue code
# and link them with any server libraries.  The Chapel compiler works
# by first translating the complete program into C and then invoking
# the system C compiler to create an executable binary. The Makefile
# created by BRAID intercepts this process after the C files have been
# generated and builds a
# <em>libtool</em>(http://www.gnu.org/software/libtool/) library
# instead. Libtool libraries contain both regular (<tt>.o</tt>) and
# position-independent (<tt>.so</tt>) versions of all the object
# files, which can be used for static and dynamic linking,
# respectively.
#
# The Chapel language already has basic support for interfacing with C
# code via the <tt>extern</tt> keyword. BRAID uses this interface as
# an entry point to open up the language for all the other languages
# supported by Babel.
#
# <h2>License</h2> 
# BRAID is released under the BSD License.
#
# <h2>Authors</h2> 
# Copyright (c) 2011, Lawrence Livermore National Security, LLC.<br/>
# Produced at the Lawrence Livermore National Laboratory.<br/>
# Written by Adrian Prantl <adrian@llnl.gov>.<br/>
# LLNL-CODE-473891. All rights reserved.<br/>
# <h3>Interns</h3>
# Shams Imam, Rice University (summer 2011)
#
# <h2>Further Reading</h2>
#
# Adrian Prantl, Thomas Epperly, Shams Imam, Vivek Sarkar<br/>
# <a href="http://pgas11.rice.edu/papers/PrantlEtAl-Chapel-Interoperability-PGAS11.pdf">
# Interfacing Chapel with Traditional HPC Programming Languages</a><br/>
# <em>PGAS 2011: Fifth Conference on Partitioned Global Address Space
# Programming Models</em>, October 2011.
#


import argparse, re, sys
import config, legal

def braid(args):
    import chapel.backend as chpl_be
    import sidl_parser, sidl_symbols, codegen

    # Babel pass-through?
    lang = args.client if args.client else args.server
    babel_langs = r'^(([cC]((\+\+)|(xx))?)|([fF]((77)|(90)|(03)))|([jJ]ava)|([pP]ython))$'
    if lang and re.match(babel_langs, lang):
        print "Invoking babel to handle language %s..." % lang
        import subprocess
        cmd = [config.BABEL_PREFIX+'/bin/babel']+sys.argv[1:]
        exit(subprocess.call(cmd))

    # No. Braid called to action!
    for sidl_file in args.sidl_files:
        sidl_ast = sidl_parser.parse(sidl_file)
        
        # SIDL operations
        if args.gen_sexp:
            print str(sidl_ast)
        if args.gen_sidl:
            print str(codegen.generate("SIDL", sidl_ast, args.debug))

        # Client code generation
        if args.client == None:
            pass
        elif re.match(r'([cC]hapel)|(chpl)', args.client):            
            sidl_ast = inject_sidl_runtime(sidl_ast, args)
            print "FIXME Handle imports here after injection of sidl runtime"
            sidl_ast, symtab = sidl_symbols.resolve(sidl_ast, args.verbose)
            chpl_be.GlueCodeGenerator(sidl_file, sidl_ast, symtab,
                                      args.makefile, args.verbose).generate_client()
        else:
            print "**ERROR: (%s) Unknown language `%s'." % (sys.argv[0], args.client)
            exit(1)

        # Server code generation
        if args.server == None:
            pass
        elif re.match(r'([cC]hapel)|(chpl)', args.server):
            sidl_ast = inject_sidl_runtime(sidl_ast, args)
            sidl_ast, symtab = sidl_symbols.resolve(sidl_ast, args.verbose)
            chpl_be.GlueCodeGenerator(sidl_file, sidl_ast, symtab,
                                      args.makefile, args.verbose).generate_server()
        else:
            print "**ERROR: (%s) Unknown language `%s'." % (sys.argv[0], args.server)
            exit(1)

def inject_sidl_runtime(sidl_ast, args):
    """
    Parse the sidl, sidlx runtime library and inject it into the
    imports field of \c sidl_ast.
    """
    import sidl_parser
    if not config.HAVE_BABEL:
        print "**ERROR: Please reconfigure %s to have Babel support." \
              %config.PACKAGE_NAME
        exit(1)

    if args.verbose:
        print "loading library 'sidl'"
    _, _, _, sidl = sidl_parser.parse(config.SIDL_PATH+'/sidl.sidl')

    if args.verbose:
        print "loading library 'sidlx'"
    _, _, _, sidlx = sidl_parser.parse(config.SIDL_PATH+'/sidlx.sidl')

    # merge in the standard library
    new_imps = [sidl, sidlx]
    # preserve other imports for further processing
    sf, req, imps, defs = sidl_ast
    for loop_imp in imps:
        scope = loop_imp[1]
        name = '.'.join(scope[1])
        if (name not in ['sidl', 'sidlx']):
            new_imps.append(loop_imp)
    
    return sf, req, new_imps, defs




if __name__ == '__main__':
    cmdline = argparse.ArgumentParser(description=config.PACKAGE_STRING+'''
- Do magically wonderful things with SIDL (scientific interface
  definition language) files.

BRAID is a high-performance language interoperability tool that generates Babel-compatible bindings for the Chapel programming language. For details on using the command-line tool, please consult the Babel manual at https://computation.llnl.gov/casc/components/ .
''',
    epilog="Please report bugs to <%s>."%config.PACKAGE_BUGREPORT)

    cmdline.add_argument('sidl_files', metavar='<file.sidl>', nargs='*',#type=file
			 help='SIDL files to use as input')

    cmdline.add_argument('--gen-sexp', action='store_true', dest='gen_sexp',
			 help='generate an s-expression')

    cmdline.add_argument('--gen-sidl', action='store_true', dest='gen_sidl',
			 help='generate SIDL output again')

    cmdline.add_argument('-c', '--client', metavar='<language>',
                         help='generate client code in the specified language'+
                         ' (Chapel, or any language supported through Babel)')

    cmdline.add_argument('-s', '--server', metavar='<language>',
                         help='generate server code in the specified language'+
                         ' (Chapel, or any language supported through Babel)')

    cmdline.add_argument('-m', '--makefile', action='store_true',
                         help='generate a default GNUmakefile')

    cmdline.add_argument('--debug', action='store_true', help='enable debugging features')
    cmdline.add_argument('--profile', action='store_true', help='enable profiling')
    cmdline.add_argument('--version', action='store_true', help='print version and exit')
    cmdline.add_argument('--license', action='store_true', help='print licensing details')
    cmdline.add_argument('-v', '--verbose', action='store_true', help='print more debug info')    

    args = cmdline.parse_args()

    if args.license:
        print config.PACKAGE_STRING
        print '''
Copyright (c) 2011, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.
Written by Adrian Prantl <adrian@llnl.gov>.
LLNL-CODE-473891. All rights reserved.

For details, see {pkg_url}
'''.format(pkg_url=config.PACKAGE_URL)
        print legal.license
        exit(0)
    if args.version:
        print config.PACKAGE_STRING
        print '# This version of BRAID was configured on %s, %s.'%(
            config.BUILD_MACHINE,config.CONFIGURE_DATE)
        exit(0)
    if len(args.sidl_files) == 0:
        cmdline.print_help()
        exit(1)

    # Dependencies
    if args.makefile and not (args.client or args.server):
        print """
Warning: The option --makefile is only effective when used in
         conjunction with --client or --server!
"""

    if args.profile:
        # Profiling
        import hotshot, hotshot.stats
        prof = hotshot.Profile('braid.prof')
        prof.runcall(braid, args)
        stats = hotshot.stats.load('braid.prof')
        stats.strip_dirs()
        stats.sort_stats('time', 'calls')
        stats.print_stats(20)
    else:
        braid(args)

    exit(0)
