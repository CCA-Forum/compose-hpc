#!/usr/bin/env python
# -*- python -*-
## @package sidl_parser
#
# This is a trivial parser for compound expressions of the form
# a(b(c),d).
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

import lex, yacc

tokens = ( 'IDENT', 'LPAREN', 'RPAREN', 'COMMA', )
t_ignore = ' \t\f' # White Space
t_IDENT  = r'\w+'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_COMMA = ','

def p_tree(p):
    '''tree : IDENT
            | IDENT LPAREN trees RPAREN'''
    if len(p) > 2: p[0] = tuple([p[1]]+p[3])
    else: p[0] = p[1]

def p_trees(p):
    '''trees : IDENT
             | IDENT COMMA trees'''
    if len(p) > 2: p[0] = [p[1]]+p[3]
    else: p[0] = [p[1]]

def t_error(t): print "**ERROR: unexpected character", t ; exit(1)
def p_error(t): print "**ERROR: malformed expression", t ; exit(1)

lexer = lex.lex(debug=0,optimize=0)
parser = yacc.yacc(optimize=0, debug=0)

def parse_tree(s):
    return parser.parse(lexer=lexer, input=s)
