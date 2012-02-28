#!/usr/bin/env python
# -*- python -*-
## @package parse_tree
#
# This is a trivial parser for compound expressions of the form
# a(b(c),d).
# It returns a python tuple.
# The expression may be closed with an optional '.'.
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

# Scanner
tokens = ( 'IDENT', 'LITERAL', 'LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET', 'COMMA' )
t_ignore = ' \r\n\t\f' # White Space
t_IDENT = r'[\w\.:]+'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_COMMA = ','

literal = r"'[^'\n\r]*'" # single-quoted literals
@lex.TOKEN(literal)
def t_LITERAL(t):
    t.value = t.value[1:-1]
    return t

# Parser
def p_start(p):
    '''start : tree IDENT
             | tree'''
    # ident is for the trailing dot
    p[0] = p[1]

def p_tree(p):
    '''tree : IDENT
            | LITERAL
            | list
            | IDENT LPAREN trees RPAREN'''
    if len(p) > 2: p[0] = tuple([p[1]]+p[3])
    else: p[0] = p[1]

def p_trees(p):
    '''trees : tree
             | tree COMMA trees'''
    if len(p) > 2: p[0] = [p[1]]+p[3]
    else: p[0] = [p[1]]

def p_list(p):
    '''list : LBRACKET RBRACKET
            | LBRACKET trees RBRACKET'''
    if p[2] == ']': p[0] = []
    else:  p[0] = p[2]

def t_error(t): print "**ERROR: unexpected character '%s'"% t.value[0] ; t.lexer.skip(1)
def p_error(t): print "**ERROR: malformed expression", t ; exit(1)

lexer = lex.lex(optimize=0, debug=0)
parser = yacc.yacc(optimize=0, debug=0)

def parse_tree(s):
    """
    Parser for compound expressions of the form
    a(b(c),[d]) and return a python tuple 
    ('a', ('b', 'c'), ['d']).
    """
    return parser.parse(lexer=lexer, input=s)

def parse_tree_file(filename):
    """
    Parser for compound expressions of the form
    a(b(c),[d]) and return a python tuple 
    ('a', ('b', 'c'), ['d']).

    \param s is expected to be a path to a file.
    """
    f = open(filename)
    data = f.read()
    f.close()
    return parser.parse(data, lexer=lexer)
