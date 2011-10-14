#!/usr/bin/env python
# -*- python -*-
## @package burg
#
# A very simple Bottom-Up Rewriting code Generator. It was inspired by
# the descriptions in
#
# - Engineering a Simple, Efficient Code Generator Generator
#   C. FRASER, D. HANSON and T PROEBSTING
#   ACM Letters on Programming Languages and Systems
#
# - Optimal Code Selection in DAGs
#   A. ERTL, POPL'99
#
# but is at this point not nearly as sophisticated.
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
import argparse, re
from parse_tree import *

def error(msg):
    print '**ERROR', msg
    print 'in line', line_number
    exit(1)

cmdline = argparse.ArgumentParser(
    description='A code generator generator',
    epilog='Please report bugs to <adrian@llnl.gov>.')

cmdline.add_argument('spec', metavar='<spec.burg>', nargs='?', type=file,
                     help='generator spec file to use as input')

cmdline.add_argument('-o', '--output', metavar='<output.py>', default='_cg.py',
                     help='file name for the output code generator')

cmdline.add_argument('--version', action='store_true', 
                     help='print version and exit')

args = cmdline.parse_args()
if args.version:  print '1.0';                 exit(0)
if not args.spec: print 'no input specified!'; exit(1)

rules = []
nonterminals = set()
line_number = 0
state = 'in_verbatim'
f = open(args.output, 'w')
for line in args.spec.readlines():
    line_number += 1
    if re.match(r'^\s*?(#.*)?$', line): # Comment
	continue

    sig = re.match(r'^%action arguments: \((.*)\)$', line)
    if sig: # Comment
	signature = sig.group(1)
	continue

    if state == 'in_verbatim':
        if re.match('^%rules:.*$', line): # Rules
            state = 'base'
            continue
        else:
            f.write(line)

    if state == 'in_rule':
	if re.match(r'^\s.*$', line):
	    f.write(line)
	    have_rule_body = True
	else:
	    if not have_rule_body:
		f.write('    pass')
	    f.write('\n')

	    state = 'base'

    if state == 'base':
	if not re.match(r'\s', line[0]):
	    # new rule

	    # we are using the following grammar:
	    # target @lang -> src @lang? : cost(c)
	    #      code
	    m = re.match('(.*)@(.+)->(.*?)(@(.+))?:cost\((\d+)\)',
			 re.sub('\s','', line))
	    if not m: error('Bad rule syntax\n'+line)

	    def chk(match, n, errmsg):
		return match.group(n) if match.group(n) \
		    else error('Syntax Error in '+errmsg)

	    # parse the components of the rule declaration
	    target      = chk(m, 1, 'target')
	    target_lang = chk(m, 2, 'target lang')
	    src         = parse_tree(chk(m, 3, 'src'))
	    src_lang    = m.group(5) if m.group(5) else target_lang
	    cost        = int(chk(m, 6, 'cost'))

	    # add the rule to our internal bookkeeping
	    action = 'action_%s_to_%s'%(str(src), target)
	    action = re.sub(r'[,.]', '_', re.sub(r"[ '\(\)]",'', action))
	    nonterminals.add(target)
	    rules.append((target, src, cost, action))

	    # print the rule action
	    f.write('# '+line)
	    if isinstance(src, tuple): # add an extra argument for each children
		args = ', '+', '.join(['a%d'%i for i in range(0, len(src))])
	    else: args = ', a0'
	    f.write('def %s(%s%s):\n'%(action, signature, args))
	    state = 'in_rule'
	    have_rule_body = False

	else: f.write('##ERR?\n')

f.write('\n')
f.write('rules = [\n')
for target, src, cost, action in rules:
    f.write('    (%s, %s, %r, %s),\n' % (target, src, cost, action))
f.write('  ]\n')
f.write('nonterminals = %r\n'%nonterminals)
f.write('\n')
f.write('def no_action(*args): pass\n')
f.write('''

# Right now the library is hard-coded and always the same. But in the
# future we could customize it by using tailored fixed-size arrays
# instead os sets to speed up the labelling process.


#print 'rules = ', repr(rules)
from parse_tree import *

def labelx(tree):
    return label1(parse_tree(tree.replace('.','_')))

def label(node):
    """
    Find a cost-minimal cover of the tree \c tree using the the rules
    defined in the global variable \c rules.

    We assume that node is a tuple containing of 
    \c (functor, child1, child2, <data>)
    """
    data = node[-1]
    if isinstance(node, str): import pdb; pdb.set_trace()
    functor = node[0]
    arity = len(node[0])
    if arity < 0: import pdb; pdb.set_trace()
    if arity > 0:
	child_labels = map(label, node[1:-1])
	# FIXME (performance) replace this with a hardcoded array
	my_labels = dict()
    else: # arity == 0
        child_labels = []
	my_labels = { node: ((node, '<terminal>', 0, no_action), 0) }

    def current_cost(target):
	for (t, src, _, action), cost in my_labels.values():
	    if target == t:
		#print "current_cost(%s) = %s"%(target,cost);
		return cost
	return 2**16

    if functor in nonterminals:
	print '**WARNING: node %r is a non-terminal symbol'%functor
	# exit(1)

    #print 'label(%s):'%str(functor)
    #print "my_labels: ", my_labels

    visited = set()
    fixpoint = False
    while not fixpoint:
	fixpoint = True
	for r in rules:
	    target, src, cost, action = r

	    #print 'src =', src
	    #import pdb; pdb.set_trace()
            if arity and not (isinstance(src, tuple)
			      and len(src) == arity
			      and (src[0] == functor 
                                or src == functor)):
                # sadly there's an ambiguity between compound types
                # and n-ary nonterminals
		continue # not compatible

	    if arity == 0:
		try:    _, basecost = my_labels[src]
		except: continue # rule does not match

	    for i in range(1, arity):
		try:    _, basecost = child_labels[src[i][0]]
		except: continue # rule does not match

	    # decide whether it pays off to add this rule
	    if cost < current_cost(target) and target not in visited:
	    #if src not in visited:
		visited.add(src)
		my_labels[target] = (r, cost)
		#print '    my_labels[',target,'] = ',(r, cost)
		fixpoint = False

    # debug output
    #for r, cost in my_labels.values():
    #    print '   ', r, ':', cost

    return tuple([my_labels]+child_labels+[data])

def reducetreex(label, target, *args):
    return reducetree1(label, target.replace('.','_'), *args)

def reducetree(label, target, *args):
    """
    Reduce a tree of labels (as generated by \c label() ) to \c target
    and execute all the code generation action associated with the
    labels as side effects.
    """
    my_labels = label[0]

    try:
        success = False
        while target in my_labels:
            r, cost = my_labels[target]
            del my_labels[target]
     
            _, target, _, action = r
            # print r, cost
            action(*tuple(list(args)+[label[-1]]))
            success = True
    except:
        pass # non-hashable

    if not success:
	print "**WARNING: no cover found!"

    for i in range(1, len(label)-1): # for each children
	reducetree(label[i], target, *args)

def codegen(src, target, *args):
    labels = label(src)
    # print 'labels = ', labels
    # print 'cost-optimal cover:'
    return reducetree(labels, target, *args)

if __name__ == '__main__':
    try:
	codegen((chpl.char, 'test'), ior.char, [], set(), '*')
	#reducetree(label(('chpl.Char')), 'ior.str', [], set())
	#print
	#reducetree(label('upcast(ior.object)'), 'ior.baseobject', [], set())
    except:
	# Invoke the post-mortem debugger
	import pdb, sys
	print sys.exc_info()
	pdb.post_mortem()

''')

print 'burg: processed %d rules, %d nonterminals'%(len(rules), len(nonterminals))
