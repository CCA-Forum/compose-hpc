#!/usr/bin/env python
# -*- python -*-
## @package parser
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

import argparse
import sidl_parser
import codegen
import chapel

if __name__ == '__main__':

    cmdline = argparse.ArgumentParser(description='''
Do magically wonderful things with SIDL (scientific interface
definition language) files.
''')
    cmdline.add_argument('sidl_files', metavar='<file.sidl>', nargs='+',#type=file
			 help='SIDL files to use as input')

    cmdline.add_argument('--gen-sexp', action='store_true', dest='gen_sexp',
			 help='generate an s-expression')

    cmdline.add_argument('--gen-sidl', action='store_true', dest='gen_sidl',
			 help='generate SIDL output again')

    cmdline.add_argument('--client', metavar='<language>',
                         help='generate client code in the specified language'+
                         ' (Chapel)')

    cmdline.add_argument('--debug', action='store_true', help='enable debugging features')

    args = cmdline.parse_args()
    for sidl_file in args.sidl_files:
	sidl_ast = sidl_parser.parse(sidl_file)

        # SIDL operations
	if args.gen_sexp:
	    print str(sidl_ast)
	if args.gen_sidl:
	    print str(codegen.generate("SIDL", sidl_ast, args.debug))

        # Client code generation
        if args.client == 'Chapel':
            try:
                chapel = chapel.Chapel(sidl_ast)
                chapel.generate_client()
            except:
                # Invoke the post-mortem debugger
                import pdb, sys
                print sys.exc_info()
                pdb.post_mortem()
        elif args.client == None:
            pass
        else:
            print "*ERROR: Unknown language `%s'." % args.client
            exit(1)
    exit(0)
