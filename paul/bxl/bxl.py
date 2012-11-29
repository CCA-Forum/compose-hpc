#!/usr/bin/env python
"""
BXL -- a minimalistic AWK work-a-like for context-free grammars.

Local variables in actions:

In a functional language we would introduce local variables like this:
   let n = y in x
In an action this can be expression using the pythonic equivalent
   (lambda n: x)(y) 

TODOs:
   * fix the ebnf grammar (see comments there)

"""

import argparse, re, sys
from pyparsing import *

__version__ = '1.0'
__license__ = 'BSD'
__author__  = 'Adrian Prantl'


re_var = re.compile(r"\$([0-9]+)")
re_bad = re.compile(r"[^0-9a-zA-Z_]")

def make_action(string):
    s = re_var.sub(r"__toks[\1-1]", string)
    try:
        printable = re_bad.sub('_',s)
        defn = '''
def action_%s(string, loc, __toks): 
    if verbose: print """action {%s}"""
    return tok_subst(__toks, (%s))
'''  %(printable, s, s) 
        #print defn
        exec defn in user_dict
        fn = user_dict['action_%s'%printable]
    except:
        print "** Syntax error in action expression:"
        print string
        print 'expanded to'
        print s
        exit(1)
    return fn

def default_action(string, loc, toks):
    #print 'default', toks
    return toks

#------------------------------------------------------------------------
# BXL public runtime library
#------------------------------------------------------------------------
user_dict = {}

exec """
def cons(a, b):
    "construct a list"
    if isinstance(b, list):
        #print 'cons', a, b, '=', [a]+b
        return [a]+b
    #FIXME
    return [a]+[b]
    #return '**TYPE ERROR: cons(%r, %r)'%(a,b)

def tok_subst(toks, val):
    '''internally used to substitue '__toks[n-1]' in the returned string 
     with the value of $n'''
    if isinstance(val, str):
        # TODO: do this in a single pass
        for i in range(len(toks)):
            val = val.replace('__toks[%d-1]'%(i+1), str(toks[i]))
    if verbose: print 'action returns', val
    return val
""" in user_dict

#------------------------------------------------------------------------
# YACC-style simplified EBNF Grammar
# definition inspired by the pyparsing EBNF example
#------------------------------------------------------------------------
all_names = '''
meta_identifier
terminal_string
regex
rule_action
syntactic_primary
definition
definitions_list
begin_rule
syntax_rule
syntax
'''.split()

symbol_table = {}
clause = {}
refs = set([])
defs = set([])
start = None

meta_identifier = Word(alphas, alphanums + '_')
terminal_string = Suppress("'") + CharsNotIn("'") + Suppress("'") ^ \
                  Suppress('"') + CharsNotIn('"') + Suppress('"')
# FIXME: this is too simple: /\//,  /\\/ won't work
regex = Suppress("/") + ZeroOrMore( CharsNotIn("/") ) + Suppress("/")
# FIXME: this is too simple: { x={} } won't work
# use nestedExpr() instead!
rule_action = Suppress('{') + ZeroOrMore( CharsNotIn("}") ) + Suppress('}')
syntactic_primary = OneOrMore( meta_identifier ^ terminal_string ^ regex )
definition = syntactic_primary + Optional( rule_action )
definitions_list = delimitedList(definition, '|')
syntax_rule = meta_identifier + Suppress('=') + definitions_list + Suppress(';')
ebnfComment = ( "/*" + ZeroOrMore( CharsNotIn("*") | ( "*" + ~Literal("/") ) ) +
                        "*/" ).streamline().setName("ebnfComment")

begin_rule = 'BEGIN' + rule_action + Suppress(';') 
syntax = Optional(begin_rule) + OneOrMore(syntax_rule)
syntax.ignore(ebnfComment)

#
#  these actions turn the BNF grammar directly into a PyParser object!
#
def do_meta_identifier(string, loc, toks):
    if toks[0] in symbol_table:
        return symbol_table[toks[0]]
    else:
        refs.add(toks[0])
        fwd = Forward()
        clause[fwd] = toks[0]
        symbol_table[toks[0]] = fwd
        return fwd

def do_terminal_string(string, loc, toks):
    return Literal(toks[0])

def do_regex(string, loc, toks):
    return Regex(toks[0])

def do_rule_action(string, loc, toks):
    return toks[0]

def do_syntactic_primary(string, loc, toks):
    toks = toks.asList()
    if len(toks) > 1:
        return And(toks)
    else:
        return [ toks[0] ]

def do_definitions_list(string, loc, toks):
    toks = toks.asList()
    if len(toks) > 1:
        # single_definition | single_definition | ...
        return Or(toks)
    else:
        # single_definition
        return [ toks[0] ]

def do_definition(string, loc, toks):
    toks = toks.asList()
    if len(toks) > 1:
        toks[0].setParseAction(make_action(toks[1]))
        return [toks[0]]
    else:
        # default action
        toks[0].setParseAction(default_action)
        return [ toks[0] ]

def do_begin_rule(string, loc, toks):
    # special AWK-style BEGIN action is always executed right away
    toks = toks.asList()
    exec toks[1] in user_dict
    return []

def do_syntax_rule(string, loc, toks):
    # regular rule
    assert toks[0].expr is None, "Duplicate definition"
    defs.add(clause[toks[0]])
    global start
    if start == None: 
        start = toks[0]
    toks[0] << toks[1]
    return [ toks[0] ]

def do_syntax(string, loc, toks):
    # syntax_rule syntax_rule ...
    return symbol_table


def parse(ebnf, given_table={}):
    '''
    main parser wrapper
    '''
    symbol_table.clear()
    symbol_table.update(given_table)
    table = syntax.parseString(ebnf)[0]

    # Consistency checks
    missing = refs - defs
    for m in missing:
        print "** missing definition for", m
    if missing: 
        exit(1)

    for name in table:
        expr = table[name]
        expr.setName(name)
        if args.verbose: expr.setDebug()
        expr.enablePackrat()
    return table

def diagnose(line, col, row):
    '''
    print a nicely formatted error message
    '''
    print line
    print ' '*(col-2)+'-^-'


#------------------------------------------------------------------------
# main
#------------------------------------------------------------------------
if __name__ == '__main__':
    # Command line argument handling
    cmdline = argparse.ArgumentParser(
        description=__doc__,
        epilog='Please report bugs to <adrian@llnl.gov>.')

    cmdline.add_argument('-f','--file', metavar='<script.bxl>', 
                         type=argparse.FileType('r'), help='BXL script')
    cmdline.add_argument('-v','--verbose', action='store_true', help='verbose mode')
    cmdline.add_argument('--version', action='store_true', help='display version number')
    cmdline.add_argument('infile', nargs='?', type=argparse.FileType('r'),
                         default=sys.stdin, help='input file')
    cmdline.add_argument('outfile', nargs='?', type=argparse.FileType('w'),
                         default=sys.stdout, help='output file')

    args = cmdline.parse_args()

    if args.version:
        print "BXL", __version__
        exit(0)

    if not args.file:  print 'no input specified!'; exit(1)

    for name in all_names:
        expr = vars()[name]
        action = vars()['do_' + name]
        expr.setName(name)
        expr.setParseAction(action)
        if args.verbose: 
            expr.setDebug()

    # load program
    try:
        if args.verbose: print "reading file", args.file.name
        g = parse(args.file.read())
    except ParseBaseException as e:
        print '** Error in %s:%d:%d'%(args.file.name, e.lineno, e.col)
        diagnose(e.line, e.col, e.lineno)
        print e
        exit(1)
    except:
        print '** Error while parsing file', args.file.name
        exit(2)

    # run program
    try:
        # redirect stdout to the outfile
        real_stdout = sys.stdout
        sys.stdout = args.outfile
        exec 'verbose = %r'%args.verbose in user_dict
        # execute the user program!
        rs = start.parseString(args.infile.read())
        for r in rs: print r

    except ParseBaseException as e:
        sys.stdout = real_stdout
        print '** Error in %s:%d:%d'%('<stdin>', e.lineno, e.col)
        diagnose(e.line, e.col, e.lineno)
        print e
        exit(1)

    except:
        sys.stdout = real_stdout
        print '** Runtime error:'
        print sys.exc_info()
        print 'launching debugger...'
        import pdb
        pdb.post_mortem()


    if args.verbose:
        print "Success!"
