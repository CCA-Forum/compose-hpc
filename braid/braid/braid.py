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
# Copyright (c) 2011, Lawrence Livermore National Security, LLC.
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
# Welcome to Braid/Babel 2!

import argparse, re
import sidl_parser, codegen, chapel, config

def braid(args):
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
            if not config.HAVE_BABEL:
                print "**ERROR: Please reconfigure BRAID to have Babel support."
                exit(1)

            print "'resolving symbol `sidl'"
            _, _, _, sidl = sidl_parser.parse(config.SIDL_PATH+'/sidl.sidl')
            print "'resolving symbol `sidlx'"
            _, _, _, sidlx = sidl_parser.parse(config.SIDL_PATH+'/sidlx.sidl')

            # merge in the standard library
            sf, req, imp, defs = sidl_ast
            sidl_ast = sf, req, imp, [sidl, sidlx, defs]
            chapel.Chapel(sidl_file, sidl_ast, args.makefile).generate_client()
        else:
            print "**ERROR: Unknown language `%s'." % args.client
            exit(1)

if __name__ == '__main__':

    cmdline = argparse.ArgumentParser(description=config.PACKAGE_STRING+'''
-- Do magically wonderful things with SIDL 
(scientific interface definition language) files.
''', epilog="Please report bugs to <%s>."%config.PACKAGE_BUGREPORT)

    cmdline.add_argument('sidl_files', metavar='<file.sidl>', nargs='*',#type=file
			 help='SIDL files to use as input')

    cmdline.add_argument('--gen-sexp', action='store_true', dest='gen_sexp',
			 help='generate an s-expression')

    cmdline.add_argument('--gen-sidl', action='store_true', dest='gen_sidl',
			 help='generate SIDL output again')

    cmdline.add_argument('-c', '--client', metavar='<language>',
                         help='generate client code in the specified language'+
                         ' (Chapel)')

    cmdline.add_argument('-m', '--makefile', action='store_true',
                         help='generate a default GNUmakefile')

    cmdline.add_argument('--debug', action='store_true', help='enable debugging features')
    cmdline.add_argument('--profile', action='store_true', help='enable profiling')
    cmdline.add_argument('--version', action='store_true', help='print version and exit')

    args = cmdline.parse_args()

    if args.version:
        print config.PACKAGE_STRING
        exit(0)
    if len(args.sidl_files) == 0:
        cmdline.print_help()
        exit(1)

    # Dependencies
    if args.makefile and not args.client:
        print """
Warning: --makefile is only effective when used in conjunction with --client!
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
