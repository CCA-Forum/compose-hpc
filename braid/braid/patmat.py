#!/usr/bin/env python
# -*- python -*-
## @package patmat
#
# Pattern matching support for s-expressions in Python.
#
# This module augments Python with a pattern-matchng syntax which
# significantly reduces the amount of typing necessary to write sidl
# code generators.
#
# The functionality is implemented in 100% Python. To use pattern
# matching in a function, the function must use the \@matcher
# decorator. In order to make it play well with editors that do syntax
# coloring and indentation, we borrowed the \c with keyword. It still
# kind of breaks flymake/pylint.
#
# \section example Example
#
# \code
# @matcher(globals()) def demo(sexp):
#    with match(sexp):
#        if ('first', Second):
#            print "found second:", Second
#        elif None:
#            print "found None"
#        elif A:
#            print "found other: ", A
#        else: raise # never reached
# \endcode
#
# The pattern matching block begins with the \code with match(expr):
# \endcode expression. In the enclosed block, the \c if and \c elif
# statements receive a new semantic, thich can be thought of a
# translation to the following form:
#
# \code
# @matcher(globals()) def demo(sexp):
#    Second = Variable()
#    A = Variable()
#    if match(sexp, ('first', Second)):
#        print "found second:", Second.binding
#    elif match(sexp, None):
#        print "found None"
#    elif match(sexp, A):
#        print "found other: ", A.binding
#    else: raise # never reached
# \endcode
#
# \li All variables starting with upper case characters are treated as
#     free variables; this is following the conventions of
#     logic-oriented programming languages.
#
# \li The underscore \c _ is treated as an anonymous variable.
#
# \li The first level of \c if and \c elif expressions under the \c
#     with \c match(expr): line are expanded to call the function \c
#     match(expr, ... .
#
# \li All occurences of upper case variables are replaced by
#     references to the values bound by those variables.
#
# \li These match blocks can be infinitely nested.
#
# This transformation is performed by the matcher decorator. The
# transformed version of the function can be compiled to a python
# source file an loaded at a later time if desired for performance
# reasons.
#
# Future plans
#
# One idea is to replace match() with an operator syntax like
# <code>a ~ b</code>.
#
# Please report bugs to <components@llnl.gov>.
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

import re, types

class Variable(object):
    """
    A logical variable for use with \c match and \c unify.
    """
    def __init__(self):
	self.binding = None

    def bind(self, value, bindings):
	"""
	Bind the variable to a \c value and record the variable onto
	the trail stack \c bindings the trail stack.

	\param value     the new value for the variable.
	\param bindings  a mutable list of bindings
	"""
	#print "binding", self, "to", value
	self.binding = value
	bindings.append(self)

    def free(self):
	"""
	\return True if this is an unbound variable.
	"""
	return self.binding == None

    def __str__(self):
        repr(self)

    def __repr__(self):
	if self.binding == None:
	    return '_'
	else:
	    return str(self.binding)


def unbind(bindings):
    """
    Remove all variable bindings recorded in \c bindings.
    \return always \c False
    """
    for var in bindings:
	var.binding = None

    bindings = []
    return False


def match(a, b):
    """
    Unify the expression \c a with the expression \c b.

    Although often conveniently used in the \c with \c match():
    syntax, this function can also be called directly. Note that match
    is commutative. \code match(a, b) == match(b, a) \endcode
    """
    return unify(a, b, [])


def expect(a, b):
    """
    Same as \c match(a,b), but raise an exception if the unification fails.
    """
    if not unify(a, b, []):
        raise Exception('type error (%s =/= %s)'%(str(a), str(b)))


def member(a, l):
    """
    Match \c a against each member of the list \c l.

    This generator will yield subsequent bindings for each element in
    \c l that can be unified with \c a.

    \todo is this a good idea? do we want something more general?
    """
    bindings = []
    for b in l:
	if unify(a, b, bindings):
	    yield True
	    unbind(bindings)

def member_chk(a, l):
    """
    True iff a is in l. Warning: Complexity is O(N).
    """
    return list(member(a, l))

def unify(a, b, bindings):
    """
    A basic unification algorithm without occurs-check

    >>> A = Variable(); B = Variable(); unify(A, B, [])
    True
    >>> A = Variable(); _ = unify(A, 1, []); A.binding
    1
    >>> A = Variable(); _ = unify((1,A), (1,2), []); A.binding
    2
    >>> A = Variable(); unify((1,2), (A,A), [])
    False
    >>> A = Variable(); B = Variable(); unify((1,(2,3),3), (1,(A,B),B), [])
    True
    """    
    type_b = type(b)
    if type_b == Variable: # Variable
	if b.binding == None:
	    b.bind(a, bindings)
	    return True
        else:
            b = b.binding
    type_a = type(a)
    if type_a == Variable: # Variable
	if a.binding == None:
	    a.bind(b, bindings)
	    return True
        else:
            a = a.binding
    if type_a == types.TupleType: # Term
	if type_b == types.TupleType: # Term
	    if len(a) != len(b):
		return unbind(bindings)
	    for i in xrange(0, len(a)):
		if not unify(a[i], b[i], bindings):
		    return False
	    return True
	else: # Atom
	    return unbind(bindings)
    else: # Atom
	if type_b == types.TupleType: # Term
	    return unbind(bindings)
	else: # Atom
	    if a == b:
		return True
	    else:
		return unbind(bindings)

# def unify_unoptimized(a, b, bindings):
#     # Unoptimized version
    
#     if isinstance(a, Variable): # Variable
#         if a.free():
#             a.bind(b, bindings)
#             return True
#         else:
#             a = a.binding
#     if isinstance(b, Variable): # Variable
#         if b.free():
#             b.bind(a, bindings)
#             return True
#         else:
#             b = b.binding
#     if isinstance(a, tuple): # Term
#         if isinstance(b, tuple): # Term
#             if len(a) != len(b):
#         	unbind(bindings)
#         	return False
#             for i in range(0, len(a)):
#         	if not unify(a[i], b[i], bindings):
#         	    unbind(bindings)
#         	    return False
#             return True
#         else: # Atom
#             unbind(bindings)
#             return False
#     else: # Atom
#         if isinstance(b, tuple): # Term
#             unbind(bindings)
#             return False
#         else: # Atom
#             if a == b:
#         	return True
#             else:
#         	unbind(bindings)
#         	return False


def unify_tuptup(a, b, bindings, length):
    """
    Optimized special case often appearing in matcher(): both \c a and
    \c b are tuples of length \c length.
    """
    for i in xrange(0, length):
        if not unify(a[i], b[i], bindings):
            unbind(bindings)
            return False
    return True

class matcher(object):
    """
    A decorator to perform the pattern matching transformation

    Usage: prefix a function definition with \c \@matcher(globals())
    """
    def __init__(self, glob, debug=False):
	"""
	\param glob   the globals dictionary to use for global variables
		      used by the function (just pass \c globals())
	\param debug  if set to \c True: print the generated code to stdout
	"""
	self.debug = debug
	self.glob = glob

    def __call__(self, f):
	fn = compile_matcher(f)
	fc = f.func_code
	if self.debug:
	    print "## "+f.__name__+':'
	    n = fc.co_firstlineno
	    for line in fn.split('\n')[n:]:
		n += 1
		print n, line
	# by using compile we can supply a filename, which results in
	# more readable backtraces
	exec(compile(fn, fc.co_filename, 'exec'), self.glob, locals())
	exec('f = %s' % f.__name__)
	# modname = '%s_matcher' % f.__name__
	# sys.path.append('.')
	# exec('import %s' % modname)
	# exec('f = %s.%s' % (modname, f.__name__))
	return f

class Counter(object):
    """
    A simple counter. Work around the fact that Python functions
    cannot modify variables in parent scopes.
    """
    def __init__(self, value=0):
	self.n = value
    def inc(self):
	self.n += 1
    def read(self):
	return self.n

def compile_matcher(f):
    """
    Compile a function f with pattern matching into a regular Python function.
    \return None.

    The function is written to a file <f.__name__>_matcher.py
    \bug not any more
    \bug 'with match()' and expect() cannot span multiple lines
    \bug string literals are not recognized by the parser
    \todo Once we are happy with the syntax, 
          rewrite this using the proper Python AST rewriting mechanisms
    """

    def indentlevel(s):
	if re.match(r'^\w*(#.*)?$', s):
	    return -1
	if re.match(r'.*\t.*', s):
	    raise Exception('Sorry... the input contains tabs:\n'+line)
	return len(re.match(r'^\s*', s).group(0))

    def tuple_len(tuple_str):
        """
        \return the arity of the tuple represented in string
        we expect the tuple to be wrapped in parentheses
        TODO: use a proper python parser for this
        """
        l = 1
        depth_par = -1 # ignore outermost parentheses
        depth_str1 = 0
        depth_str2 = 0
        valid = False
        for char in tuple_str:
            if   char == ',' and depth_str1+depth_str2 == 0 and depth_par == 0: l += 1
            elif char == '(' and depth_str1+depth_str2 == 0: depth_par += 1; valid = True
            elif char == ')' and depth_str1+depth_str2 == 0: depth_par -= 1
            elif char == "'" and depth_str2 == 0: 
                if depth_str1 == 0: depth_str1 = 1
                else: depth_str1 = 0
            elif char == '"' and depth_str1 == 0:
                if depth_str2 == 0: depth_str2 = 1
                else: depth_str2 = 0
        assert(depth_par+depth_str1+depth_str2 == -1)
        if not valid: return -1
        return l

    def scan_variables(rexpr):
	"""
	extract variable names from the right-hand side expression
	and rename anonymous variables to something unique
        modifies regalloc[], anonymous_vars
	"""
	reserved_words = r'(\..*)|(False)|(True)|(None)|(NotImplemented)|(Ellipsis)'
	matches = re.findall(r'(^|\W)([_A-Z]\w*)($|[^\(\w])', rexpr)
	names = set([])
	for m in matches:
	    var = m[1]
	    if m[0] == '.' or re.match(reserved_words, var):
		# ignore reserved words
		continue

	    if var[0] == '_': # name starts with underscore
		# Generate names for anonymous variables
		anonymous_vars.inc()
		var1 = '_G%d'%anonymous_vars.read()
		rexpr = re.sub(r'(^|\W)%s($|[^\(\w])'%var,
			       r'\1%s\2'%var1,
			       rexpr, 1)
		var = var1

	    # Remember variable name if it is new
	    if var not in names:
		regalloc[-1].append(var)
		names.add(var)

	numregs[-1] = max(numregs[-1], len(names))
	return rexpr

    def get_vardecls():
        decls = []
        for i in range(0, numregs[-1]):
            d = '' # FIXME
            decls.append('_reg%s%d = %sVariable()' % (d, i, patmat_prefix))
        return decls

    def substitute_registers(line, d):
        """
        replace variables with registers in line
        """
        for i in range(0, len(regalloc[-1])):
            line = re.sub(r'(\W|^)'+regalloc[-1][i]+r'(\W|$)',
                          r'\1_reg%s%d\2' % (d, i),
                          line)
        return line


    def depthstr(n):
	"""generate unique register names for each nesting level"""
	if n == 0:
	    return ""
	else: return chr(n+ord('a'))

    def append_line(line):
        if bucket <> None:
            bucket.append('    '+line[base_indent:])
        else:
            num_lines.inc()
            dest.append(line[base_indent:])

    def insert_line(pos, line):
	num_lines.inc()
	dest.insert(pos, line[base_indent:])

    fc = f.func_code
    # access the original source code
    src = open(fc.co_filename, "r").readlines()

    # get the indentation level of the first nonempty line
    while True:
	base_indent = indentlevel(src[fc.co_firstlineno])
	if base_indent > -1:
	    break

    # imports = ', '.join(
    #     filter(lambda s: s <> f.__name__, parse_globals(src, base_indent)))

    dest = []
    # assign a new function name
    # m = re.match(r'^def +([a-zA-Z_][a-zA-Z_0-9]*)(\(.*:) *$', dest[0])
    # funcname = m.group(1)
    # funcparms = m.group(2)
    dest.append("""
#!/usr/env python
# This file was automatically generated by the @matcher decorator. Do not edit.
# module %s_matcher
# from patmat import matcher, Variable, match
""" % f.__name__)

    # match line numbers with original source
    dest.append('\n'*(fc.co_firstlineno-5))
    num_lines = Counter(3) # number of elements in dest
    anonymous_vars = Counter(0) # number of anonymous variables
    n = fc.co_firstlineno
    while re.match(r'^\s*@', src[n]):
        n += 1 # skip decorators
    
    bucket = None
    append_line(src[n])

    # FIXME: wouldn't one stack with a tuple/class of these be nicer?
    # and even better: couldn't we just recurse when we encounter a with block?

    # stacks
    lexpr = [] # lhs expr of current match block
    numregs = [] # number of simulatneously live variables
    buckets = [] # sorting buckets for the matching rules
    regalloc = [] # associating variables with registers
    withbegin = [] # beginning of current with block
    withindent = [] # indent level of current with block
    matchindent = [] # indent level of current match block
    patmat_prefix = ''
    for line in src[n+1:]+['<<EOF>>']:
	# append_line('# %s:%s\n' % (fc.co_filename, n+fc.co_firstlineno))

	il = indentlevel(line)
	# check for empty/comment-only line (they mess with the indentation)
	if re.match(r'^\s*(#.*)*$', line):
	    # make sure it still is a comment after shifting the line
	    # to the left
	    line = "#"*base_indent+line
	    append_line(line)
	    continue

	# leaving a with block
	while len(withindent) > 0 and il <= withindent[-1]:
	    # insert registers declarations
	    decls = get_vardecls()

	    # put all vardecls in one line, so we don't mess with the line numbering
	    insert_line(withbegin[-1],
			' '*(withindent[-1]) + '; '.join(decls) + '\n')

            # output the rules in decreasing order
            insert_line(withbegin[-1], ' '*(withindent[-1])+
                        '_len = len({lexpr}) if isinstance({lexpr}, tuple) else -1\n'
                        .format(lexpr=lexpr[-1]))
            
            ind = ' '*(withindent[-1]-base_indent)
            s = [ind+'_NO_MATCH = False\n']
            else_notmatch = ' '*(withindent[-1]-base_indent+4)+ \
                'else: _NO_MATCH = True\n'
            first = True
            for l, rules in reversed(sorted(buckets[-1].items())):
                if not first: s.append(else_notmatch)
                s.append(ind+'%sif _len == %d:\n'%('' if first else 'el', l))
                s.extend(rules)
                first = False
            if not first: # had at least 1
                s.append(else_notmatch)

            s.append(ind+'else: _NO_MATCH = True\n')
            s.append(ind+'if not _NO_MATCH: pass\n')
            dest.insert(withbegin[-1]+2, ''.join(s))

            # cleanup
            buckets.pop()
	    matchindent.pop()
	    withindent.pop()
	    withbegin.pop()
	    regalloc.pop()
	    numregs.pop()
	    lexpr.pop()
	    if len(withindent) <> len(matchindent):
		raise Exception(
                    '**ERROR: %s:%d: missing if statement inside of if block'%
		    (fc.co_filename, fc.co_firstlineno+2+num_lines.read()))
	    # ... repeat for all closing blocks

	# end of function definition
	if il <= base_indent:
	    break

	# entering a with block
	m = re.match(r'^ +with +(patmat\.)?match\((.*)\) *: *$', line)
	if m:
	    if m.group(1):
		patmat_prefix = m.group(1)
	    lexpr.append(m.group(2))
	    numregs.append(0)
	    regalloc.append([])
	    withindent.append(il)
	    withbegin.append(num_lines.read()-1)
	    line = ''
            bucket = None
            buckets.append(dict())

	# expect() is handled completely in here
        # putting expect here is still half-baked...
        # there is no automatic replacement of named variables and
        # that is inconsistent with the with syntax
	m = re.match(r'^ +(patmat\.)?expect\((.*)\) *$', line)
	if m:
	    if m.group(1):
		patmat_prefix = m.group(1)
	    numregs.append(0)
	    regalloc.append([])
            exprs = scan_variables(m.group(2))
	    # put all in one line, so we don't mess with the line numbering
	    vardecls = '; '.join(get_vardecls())
            if len(vardecls) > 0:
                vardecls += '; '
            line = substitute_registers(
                ' '*il+vardecls+patmat_prefix+'expect(%s)'%exprs+'\n', '')
            regalloc.pop()
	    numregs.pop()

	# inside a matching rule
	if len(lexpr) > 0:
	    skip_blanks = False
	    # record the current indentation
	    if len(withindent) <> len(matchindent):
		if re.match(r'^ *if', line):
		    matchindent.append(il)
		else:
		    skip_blanks = True

	    if not skip_blanks:
		# remove one layer of indentation
		leftshift = matchindent[-1]-withindent[-1]
		line = line[leftshift:]
		matchind = matchindent[-1]-leftshift

		# match if() / elif()
		m = re.match(r'^('+' '*matchind+r')((el)?if) +([^:]*):(.*)$', line)

		if m:
                    el_if = m.group(1)
		    rexpr = m.group(4)

                    # we are sorting the match operations by tuple length
                    tl = tuple_len(rexpr)
                    if tl > 1: 
                        # prepare a new sorting bucket
                        if tl not in buckets[-1].keys():
                            buckets[-1][tl] = []
                            el_if = 'if'
                        else:
                            el_if = 'elif'
                        bucket = buckets[-1][tl]

                        # the optimized matching rule
                        regalloc[-1] = []
                        line = '%s%s %sunify_tuptup(%s, %s, [], %d):' \
                            % (m.group(1), el_if,
                               patmat_prefix,
                               lexpr[-1],
                               scan_variables(rexpr),
                               tl)

                    else: 
                        bucket = None
                        el_if = 'elif'

                        # the generic matching rule
                        regalloc[-1] = []
                        line = '%s%s %sunify(%s, %s, []):' \
                            % (m.group(1), el_if,
                               patmat_prefix,
                               lexpr[-1],
                               scan_variables(rexpr))

		    # substitute registers for variables
                    line = substitute_registers(line, depthstr(len(lexpr)-1))
		    # split off the part behind the ':' and append it
		    then = m.group(5)
		    if len(then) > 0:
			append_line(line)
			line = ' '*il+then
		    line += '\n'
                else:
                    if re.match(r'^'+' '*matchind+r'else.*$', line):
                        bucket = None


	# every time
	if len(withbegin) > 0:
            # substitute registers for variables
            # ... can be done more efficiently
            j = 0
            for alloc in regalloc:
                d = depthstr(j)
                for i in range(0, len(alloc)):
                    line = re.sub(r'(\W|^)'+alloc[i]+r'(\W|$)',
                                  r'\1_reg%s%d.binding\2' % (d, i), line)
                j += 1

	# copy the line to the output
	append_line(line)

    #modname = '%s_matcher' % f.__name__
    #f = open(modname+'.py', "w")
    buf = "".join(dest)
    #f.write(buf)
    #f.close()
    return buf
