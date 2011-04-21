#!/usr/bin/env python
# -*- python -*-
## @package codegen
# Several code generators.
#
# General design principles (a.k.a. lessons learned from Babel code generators
#
# * It is ok to duplicate code if it increases locality and thus
#   improves readability
#
# Each code generator is split into two components: the actual code
# generator and the <em>language</em>File class which implements
# things such as indentation and line wrapping. The code generator
# implements a \c generate(sexpr) function which takes an \c ir node
# in s-expression form.
#
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

import sys, re
import ir, sidl
from patmat import matcher, Variable, match, member

languages = ["C", "CXX", "F77", "F90", "F03", "Python", "Java"]

def generate(language, ir_code, debug=False):
    """
    Call the appropriate generate() function.  

    \param language One of
    \c ["C", "CXX", "F77", "F90", "F03", "Python", "Java"]
    
    \param ir_code  Intermediate representation input.
    \param debug    Turn on patmat debugging.
    \return         string

    >>> generate('C', ir.Plus(1, 2))
    '1 + 2'
    >>> [generate(lang, 1) for lang in languages] 
    ['1', '1', '1', '1', '1', '1', '1']
    >>> [generate(lang, ir.Plus(1, 2)) for lang in languages] 
    ['1 + 2', '1 + 2', '1 + 2', '1 + 2', '1 + 2', '1 + 2', '1 + 2']
    """
    # apparently CPython does not implement proper tail recursion
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 2**16))

    try:
        if language == "C": 
            return str(CCodeGenerator().generate(ir_code, CFile()))

        elif language == "CXX": 
            return str(CXXCodeGenerator().generate(ir_code, CXXFile()))

        elif language == "F77": 
            return str(Fortran77CodeGenerator().generate(ir_code, F77File()))

        elif language == "F90": 
            return str(Fortran90CodeGenerator().generate(ir_code, F90File()))

        elif language == "F03": 
            return str(Fortran03CodeGenerator().generate(ir_code, F03File()))

        elif language == "Python": 
            return str(PythonCodeGenerator().generate(ir_code, PythonFile()))

        elif language == "Java": 
            return str(JavaCodeGenerator().generate(ir_code, JavaFile()))

        elif language == "SIDL": 
            return str(SIDLCodeGenerator().generate(ir_code, SIDLFile()))

        else: raise Exception("unknown language")
    except:
        # Invoke the post-mortem debugger
        import pdb
        print sys.exc_info()
        print sys.exc_info()
        if debug:
            pdb.post_mortem()
        else: 
            exit(1)

def generator(fn):
    """
    sanity-check decorator for generator functions
    """
    def wrapped(*args):
        r = fn(*args)
        if r == None or isinstance(r, tuple):
            print args
            print '---->', r
            raise Exception("Code generator output failed sanity check")
        return r

    return wrapped

def accepts(*types):
    """
    Enforce function argument types. 
    Taken from directly from pep-0318.
    """
    def check_accepts(f):
        assert len(types) == f.func_code.co_argcount
        def new_f(*args, **kwds):
            for (a, t) in zip(args, types):
                assert isinstance(a, t), \
                       "arg %r does not match %s" % (a,t)
            return f(*args, **kwds)
        new_f.func_name = f.func_name
        return new_f
    return check_accepts

def returns(rtype):
    """
    Enforce function return types. 
    Taken from directly from pep-0318.
    """
    def check_returns(f):
        def new_f(*args, **kwds):
            result = f(*args, **kwds)
            assert isinstance(result, rtype), \
                   "return value %r does not match %s" % (result,rtype)
            return result
        new_f.func_name = f.func_name
        return new_f
    return check_returns

def sep_by(separator, strings):
    """
    Similar to \c string.join() but appends the seperator also after
    the last element, if any.
    """
    r = separator.join(strings)
    if len(strings) > 0:
        return r+separator
    else:
        return r


class Scope(object):
    """
    This class provides an interface to the output file that is better
    suited for generating source code than a sequential output
    stream. In particular, it provides a solution for the problem that
    often we want to add another definition the beginning of an
    enclosing scope.

    Scopes are nested and generally look like this:

    <pre>
    +--------------------+     +-----------------------------------+
    | header             |  +  | defs                              |
    +--------------------+     +-----------------------------------+
    </pre>

    Subclasses of this might offer more sophisticated layouts.


    \c self.indent_level is the level of indentation used by this \c
    Scope object. The \c indent_level is constant for each \c Scope
    object. If you want to change the indentation, the idea is to
    create a child \c Scope object with a different indentation.

    """
    def __init__(self, 
                 parent=None, 
                 relative_indent=0, 
                 separator='\n',
                 max_line_length=80):
        """
        \param parent           The enclosing scope.

        \param relative_indent  The amount of indentation relative to
                                the enclosing scope.

        \param separator        This string will be inserted between every
                                two definitions.

        \param max_line_length  The maximum length that a line should occupy.
                                Longer lines will be broken into shorter ones
                                automatically.
        """
        self.parent = parent
        self._header = []
        self._defs = []
        self._pre_defs = []
        self._post_defs = []
        self.relative_indent = relative_indent
        self._max_line_length = max_line_length
        if parent: self.indent_level = parent.indent_level + relative_indent
        else:      self.indent_level = relative_indent
        self._sep = separator+' '*self.indent_level

    def has_declaration_section(self):
        """
        \return whether this scope has a section for variable declarations.
        """
        return False

    def new_header_def(self, s):
        """
        append definition \c s to the header of the scope
        """
        self._header.append(self.break_line(s))

    def new_def(self, s):
        """
        Append definition \c s to the scope. Also adds anything
        previously recorded by \c pre_def or \c post_def.  For
        convenience reasons it returns \c self, see the code
        generators on examples why this is useful

        \return \c self
        """
        #print 'new_def', s
        if s <> self:
            self._defs.extend(self._pre_defs)
            self._defs.append(self.break_line(str(s)))
            self._defs.extend(self._post_defs)
            self._pre_defs = []
            self._post_defs = []
        return self

    def pre_def(self, s):
        """
        Record a definition \c s to be added to \c defs before the
        next call of \c new_def.
        """
        self._pre_defs.append(self.break_line(s))

    def post_def(self, s):
        """
        Record a definition \c s to be added to \c defs after the
        next call of \c new_def.
        """
        self._post_defs.append(self.break_line(s))

    def get_defs(self):
        """
        return a list of all definitions in the scope
        """
        return self._header+self._defs

    def __str__(self):
        """
        Perform the actual translation into a readable string,
        complete with indentation and newlines.
        """
        #print self._header, '+', self._defs, 'sep="',self._sep,'"'
        #import pdb; pdb.set_trace()
        s = self._sep.join(self._header) + self._sep.join(self._defs)
        if len(s) > 0:
            return ' '*self.relative_indent + s
        return s

    def break_line(self, string):
        """
        Break a string of C-like code at max_line_length.
        """
        # FIXME: this can't stay this way. We should be doing this only once per line
        lines = []
        for ln in string.split('\n'):
            tokens = ln.split(' ')
            while len(tokens) > 0:
                line = ""
                while (len(tokens) > 0 and 
                       len(line)+len(tokens[0]) < self._max_line_length):
                    line += tokens.pop(0)
                    if len(tokens): line += ' '
                lines += [line]

        il = self.indent_level + max(self.relative_indent, 2) # toplevel
        indent = '\n' +' '*il
        return indent.join(lines)

class SourceFile(Scope):
    """
    This class represents a generic source file
    """
    def __init__(self, parent=None, relative_indent=0, separator='\n'):
        super(SourceFile, self).__init__(
            parent, relative_indent, separator)

    def has_declaration_section(self):
        return True

    def __str__(self):
        """
        Perform the actual translation into a readable string,
        complete with indentation and newlines.
        """
        #print self._header, '+', self._defs, 'sep="',self._sep,'"'
        #import pdb; pdb.set_trace()
        return (' '*self.relative_indent
                +self._sep.join( self._header+self._defs ))


class Function(Scope):
    """
    This class represents a function/procedure
    """
    def has_declaration_section(self):
        return True

class GenericCodeGenerator(object):
    """
    All code generators shall implement this interface and inherit
    from this class.

    Classes inheriting from this one are expected to provide
    type_map, un_op and bin_op.
    """
    
    @generator
    @matcher(globals(), debug=False)
    def generate(self, node, scope=SourceFile()):
        """
        Language-independent generator rules

        \param node       s-expression-based intermediate representation (input)
        \param scope      the \c Scope object the output will be written to
        \return           a string containing the expression for \c node
        """
        def gen(node):
            return self.generate(node, scope)

        with match(node):
            if (ir.stmt, Expr):
                return scope.new_def(gen(Expr))

            elif (ir.infix_expr, Op, A, B): return ' '.join((gen(A), self.bin_op[Op], gen(B)))
            elif (ir.prefix_expr, Op, A):   return ' '.join((self.un_op[Op], gen(A)))
            elif (ir.primitive_type, T):    return self.type_map[T]
            elif (ir.struct_item, Type, Name): return Name
            elif (A):        
                if (isinstance(A, list)):
                    for defn in A:
                        scope.new_def(gen(defn))
                elif (isinstance(A, int)):   return str(A)
                elif (isinstance(A, float)): return str(A)
                elif (isinstance(A, complex)): return str(A)
                elif (isinstance(A, long)): return str(A)
                elif (isinstance(A, str)): 
                    #print "FIXME: string `%s' encountered. Fix your generator"%A
                    return A
                elif (isinstance(A, tuple)):
                    raise Exception("not implemented: "+repr(A))
                elif (A == None):
                    raise Exception("None encountered "+repr(A))
                else:
                    raise Exception("unexpected type"+repr(A))
            else: raise Exception("match error: "+repr(node))
        return scope

    def gen_in_scope(self, defs, child_scope):
        """
        building block for things like \c gen_comma_sep
        """
        r = self.generate(defs, child_scope)
        if (isinstance(r, str)):
            raise Exception("unexpected retval")
        return str(child_scope)

    def get_item_type(self, struct, item):
        """
        \return the type of the item named \c item
        """
        _, (_, _, _, items), _ = struct
        Type = Variable()
        for _ in member((ir.struct_item, Type, item), items):
            return Type.binding
        raise Exception("Struct has no member "+item)

# ----------------------------------------------------------------------
# C      FORTRAN 77
# ----------------------------------------------------------------------
class F77File(SourceFile):
    """
    This class represents a Fortran 77 source file
    """

    def __init__(self, parent=None, relative_indent=0):
        super(F77File, self).__init__(
            parent=parent,
            relative_indent=relative_indent)
        self.label = 0

    def new_def(self, s, indent=0):
        """
        Append definition \c s to the scope
        \return  \c self
        """
        # break long lines
        tokens = s.split()
        line = ' '*(self.relative_indent+indent)
        while len(tokens) > 0: 
            while (len(tokens) > 0 and 
                   len(line)+len(tokens[0]) < 62):
                line += tokens.pop(0)+' '
            super(F77File, self).new_def(line)
            line = '&' # continuation character
        return self

    def new_label(self):
        """
        Create a new label before the current definition.
        """
        self.label += 10
        l = self.label
        self.pre_def('@%3d'%l)
        return l

    def __str__(self):
        """
        Perform the actual translation into a readable string,
        complete with indentation and newlines.
        """
        #print self._header, '+', self._defs, 'sep="',self._sep,'"'
        #import pdb; pdb.set_trace()        
        data = ''
        label = False
        for defn in self.get_defs():
            if label: 
                label = False
                data += defn+'\n'            
            elif defn[0] == '&': data += '     &      '+defn[1:]+'\n'
            elif defn[0] == '@':       
                   label = True; data += ' %s    '% defn[1:]
            else:                data += '        '+defn+'\n'

        return data

class F77Scope(F77File):
    """Represents a list of statements in an indented block"""
    def __init__(self, parent):
        super(F77Scope, self).__init__(
            parent,
            relative_indent=parent.relative_indent+2)
        self._defs = [''] # start on a new line
    
    def has_declaration_section(self):
        return False

class Fortran77CodeGenerator(GenericCodeGenerator):
    type_map = {
        'void':        "void",
        'bool':        "logical",
        'character':   "character",
        'dcomplex':    "double complex",
        'double':      "double precision",
        'fcomplex':    "complex",
        'float':       "real",
        'int':         "integer*4",
        'long':        "integer*8",
        'opaque':      "integer*8",
        'string':      "character*256",
        'enum':        "integer*8",
        'struct':      "integer*8",
        'class':       "integer*8",
        'interface':   "integer*8",
        'package':     "void",
        'symbol':      "integer*8"
        }

    bin_op = {
        'log_or':  '.or.',
        'log_and': '.and.',
        'eq':      '.eq.',
        'ne':      '.neq.',
        'bit_or':  '|',
        'bit_and': '&',
        'bit_xor': '^',
        'lt':      '<',
        'gt':      '>',
        'lshift':  '<<',
        'rshift':  '>>',
        'plus':    '+',
        'minus':   '-',
        'times':   '*',
        'divide':  '/',
        'modulo':  '%',
        'rem':     'rem',
        'pow':     'pow'
        }

    un_op = { 
        'log_not': '.not.',
        'bit_not': '~'
        }

    @generator
    @matcher(globals(), debug=False)
    def generate(self, node, scope):
        """
        Fortran 77 code generator

        \param node         sexp-based intermediate representation (input)
        \param scope  the Scope object the output will be written to
        """
        # recursion
        def gen(node):
            return self.generate(node, scope)

        def declare_var(typ, name):
            """
            add a declaration for a variable to the innermost scope if
            such a declaration is not already there
            """
            s = scope
            while not s.has_declaration_section():
                s = s.parent
            decl = self.get_type(typ)+' '+gen(name)
            if list(member(decl, s._header)) == []:
                s.new_header_def(decl)

        def new_def(s):
            # print "new_def", s
            return scope.new_def(s)

        def pre_def(s):
            # print "pre_def", s
            return scope.pre_def(s)

        def new_scope(prefix, body, suffix=''):
            '''used for things like if, for, ...'''
            s = F77Scope(parent=scope)
            new_def(prefix)
            self.generate(body, s)
            # copy all defs into the F77File which takes care of the
            # F77's weird indentation rules
            # FIXME get rid of these side effects.. this is just asking for trouble
            for defn in s.get_defs():
                scope.new_def(defn, s.relative_indent)
            new_def(suffix)
            return scope

        with match(node):
            if (ir.return_, Expr):
                return "retval = %s" % gen(Expr)

            elif (ir.primitive_type, T): return type_map[T]
            elif (ir.struct, (Package), (Name), _): 
                return ("%s_%s"%(Package, Name)).lower()

            elif (ir.get_struct_item, Struct, Name, Item):
                tmp = 'tmp_%s_%s'%(gen(Name), gen(Item))
                declare_var(self.get_item_type(Struct, Item), tmp)
                _, type_, _ = Struct
                pre_def('call %s_get_%s_f(%s, %s)' % (
                        gen(self.get_type(type_)), gen(Item), gen(Name), tmp))
                return tmp

            elif (ir.set_struct_item, (_, Struct, _), Name, Item, Value):
                return 'call %s_set_%s_f(%s, %s)' % (
                    gen(self.get_type(Struct)), gen(Item), gen(Name), gen(Value))

            elif (ir.fn_defn, ir.void, Name, Attrs, Args, Excepts, From, Requires, Ensures, Body):
                return new_def('''
                subroutine %s
                  %s
                  %s
                end subroutine %s
                ''' % (Name, gen(Args),
                       gen(FunctionScope(scope), Body), Name))

            elif (ir.fn_defn, Typ, Name, Attrs, Args, Excepts, Froms, Requires, Ensures):
                return new_def('''
                subroutine %s
                  %s
                  %s,
                  retval
                end function %s
            ''' % (Typ, Name, gen(Args),
                   gen(FunctionScope(scope), Body), Name))

            elif (ir.do_while, Condition, Body):
                label = scope.new_label()
                gen(Body)
                return new_scope('if (%s) then'%gen(Condition), 
                                      (ir.stmt, (ir.goto, str(label))), 
                                      'end if')

            elif (ir.if_, Condition, Body):
                return new_scope('if (%s) then'%gen(Condition), Body, 'end if')

            elif (ir.var_decl, Type, Name): declare_var(Type, gen(Name))
            elif (ir.goto, Label):    return 'goto '+Label
            elif (ir.assignment):     return '='
            elif (ir.true):           return '.true.'
            elif (ir.false):          return '.false.'
            elif (Expr):
                return super(Fortran77CodeGenerator, self).generate(Expr, scope)

            else: raise Exception("match error")

# ----------------------------------------------------------------------
# Fortran 90
# ----------------------------------------------------------------------
class F90File(SourceFile):
    """
    This class represents a Fortran 90 source file
    """
    def __init__(self,parent=None,relative_indent=2):
        super(F90File, self).__init__(parent,relative_indent)

    def new_def(self, s, indent=0):
        """
        Append definition \c s to the scope
        \return  \c self
        """
        # split long lines
        tokens = s.split()
        while len(tokens) > 0: 
            line = ' '*(self.relative_indent+indent)
            while (len(tokens) > 0 and 
                   len(line)+len(tokens[0]) < 62):
                line += tokens.pop(0)+' '
            
            if len(tokens) > 0:
                line += '&'
            super(F90File, self).new_def(line)
        return self


    def __str__(self):
        """
        Perform the actual translation into a readable string,
        complete with indentation and newlines.
        """
        return '\n'.join(self._header+self._defs)+'\n'

class F90Scope(F90File):
    """Represents a list of statements in an indented block"""
    def __init__(self, parent):
        super(F90Scope, self).__init__(
            parent,
            relative_indent=parent.relative_indent+2)
        self._defs = [''] # start on a new line
    
    def has_declaration_section(self):
        return False

class Fortran90CodeGenerator(GenericCodeGenerator):
    """
    Fortran 90 code generator
    """
    bin_op = {
        'log_or':  '.or.',
        'log_and': '.and.',
        'eq':      '.eq.',
        'ne':      '.neq.',
        'bit_or':  '|',
        'bit_and': '&',
        'bit_xor': '^',
        'lt':      '<',
        'gt':      '>',
        'lshift':  '<<',
        'rshift':  '>>',
        'plus':    '+',
        'minus':   '-',
        'times':   '*',
        'divide':  '/',
        'modulo':  '%',
        'rem':     'rem',
        'pow':     'pow'
        }

    un_op = { 
        'log_not': '.not.',
        'bit_not': '~'
        }

    @matcher(globals(), debug=False)
    def get_type(self, node):
        """\return a string with the type of the IR node \c node."""
        with match(node):
            if ('struct', Type, _): return Type
            elif ('void'):        return "void"
            elif ('bool'):        return "logical"
            elif ('character'):   return "character (len=1)"
            elif ('dcomplex'):    return "complex (kind=sidl_dcomplex)"
            elif ('double'):      return "real (kind=sidl_double)"
            elif ('fcomplex'):    return "complex (kind=sidl_fcomplex)"
            elif ('float'):       return "real (kind=sidl_float)"
            elif ('int'):         return "integer (kind=sidl_int)"
            elif ('long'):        return "integer (kind=sidl_long)"
            elif ('opaque'):      return "integer (kind=sidl_opaque)"
            elif ('string'):      return "character (len=*)"
            elif ('enum'):        return "integer (kind=sidl_enum)"
            elif ('struct'):      return ""
            elif ('class'):       return ""
            elif ('interface'):   return ""
            elif ('package'):     return ""
            elif ('symbol'):      return ""
            else: return super(Fortran90CodeGenerator, self).get_type(node)

    @generator
    @matcher(globals(), debug=False)
    def generate(self, node, scope=F90File()):
        # recursion
        def gen(node):
            return self.generate(node, scope)

        def declare_var(typ, name):
            s = scope
            while not s.has_declaration_section():
                s = s.parent
            s.new_header_def(self.get_type(typ)+' '+gen(name))

        def new_scope(prefix, body, suffix=''):
            '''used for things like if, for, ...'''
            s = F90Scope(parent=scope)
            new_def(prefix)
            self.generate(body, s)
            # copy all defs into the F90File which takes care of the
            # F90's weird indentation rules
            # FIXME! should really be the file! not scope (messes with
            # indentation otherwise)
            for defn in s.get_defs():
                scope.new_def(defn, s.relative_indent)
            new_def(suffix)
            return scope

        def new_def(s):
            # print "new_def", s
            return scope.new_def(s)

        def pre_def(s):
            # print "pre_def", s
            return scope.pre_def(s)

        with match(node):
            if (ir.stmt, Expr):
                return scope.new_def(gen(Expr)+'\n')

            if ('return', Expr):
                return "retval = %s" % gen(Expr)

            elif (ir.get_struct_item, _, Name, Item):
                return gen(Name)+'%'+gen(Item)

            elif (ir.set_struct_item, _, Name, Item, Value):
                return gen(Name)+'%'+gen(Item)+' = '+gen(Value)

            elif (ir.fn_defn, ir.void, Name, Attrs, Args, Excepts, Froms, Requires, Ensures, Body):
                return '''
                subroutine %s
                  %s
                  %s
                end subroutine %s
                ''' % (Name, gen(Args), gen(Body), Name)

            elif (ir.fn_defn, Typ, Name, Attrs, Args, Excepts, Froms, Requires, Ensures):
                return '''
                function %s
                  %s
                  %s
                end function %s
            ''' % (Typ, Name, gen(Args), gen(Body), Name)
            elif (ir.do_while, Condition, Body):
                new_scope('do', Body, '  if (.not.%s) exit'%gen(Condition))
                new_def('end do')
                return scope
            elif (ir.if_, Condition, Body):
                return new_scope('if (%s) then'%gen(Condition), Body, 'end if')
            elif (ir.var_decl, Type, Name): declare_var(gen(Type), gen(Name))
            elif (ir.assignment):     return '='
            elif (ir.eq):             return '.eq.'
            elif (ir.true):           return '.true.'
            elif (ir.false):          return '.false.'
            elif (Expr):
                return super(Fortran90CodeGenerator, self).generate(Expr, scope)

            else: raise Exception("match error")

# ----------------------------------------------------------------------
# Fortran 2003
# ----------------------------------------------------------------------
class F03File(F90File):
    """
    This class represents a Fortran 03 source file
    """
    def __init__(self):
        super(F03File, self).__init__(relative_indent=4)

class Fortran03CodeGenerator(Fortran90CodeGenerator):
    """
    Fortran 2003 code generator
    """
    @matcher(globals(), debug=False)
    def get_type(self, node):
        """\return a string with the type of the IR node \c node."""
        with match(node):
            if ('struct', Type, _): return Type
            elif ('void'):        return "void"
            elif ('bool'):        return "logical"
            elif ('character'):   return "character (len=1)"
            elif ('dcomplex'):    return "complex (kind=sidl_dcomplex)"
            elif ('double'):      return "real (kind=sidl_double)"
            elif ('fcomplex'):    return "complex (kind=sidl_fcomplex)"
            elif ('float'):       return "real (kind=sidl_float)"
            elif ('int'):         return "integer (kind=sidl_int)"
            elif ('long'):        return "integer (kind=sidl_long)"
            elif ('opaque'):      return "integer (kind=sidl_opaque)"
            elif ('string'):      return "character (len=*)"
            elif ('enum'):        return "integer (kind=sidl_enum)"
            elif ('struct'):      return ""
            elif ('class'):       return ""
            elif ('interface'):   return ""
            elif ('package'):     return ""
            elif ('symbol'):      return ""
            else: return super(Fortran03CodeGenerator, self).get_type(node)

    """
    Struct members: These types do not need to be accessed via a function call.
    """
    struct_direct_access = ['dcomplex', 'double', 'fcomplex', 'float', 
                            'int', 'long', 'enum']

    @generator
    @matcher(globals(), debug=False)
    def generate(self, node, scope=F03File()):
        # recursion
        def gen(node):
            return self.generate(node, scope)

        def declare_var(typ, name):
            s = scope
            while not s.has_declaration_section():
                s = s.parent
            s.new_header_def(self.get_type(typ)+' '+gen(name))

        def new_def(s):
            # print "new_def", s
            return scope.new_def(s)

        def pre_def(s):
            # print "pre_def", s
            return scope.pre_def(s)

        with match(node):
            if ('return', Expr):
                return "retval = %s" % gen(Expr)

            elif (ir.get_struct_item, Struct, Name, Item):
                t = self.get_item_type(Struct, Item)
                if t in self.struct_direct_access:
                    return gen(Name)+'%'+gen(Item)
                else:
                    return 'get_'+gen(Item)+'('+gen(Name)+')'

            elif (ir.set_struct_item, Struct, Name, Item, Value):
                t = self.get_item_type(Struct, Item)
                if t in self.struct_direct_access:
                    return gen(Name)+'%'+gen(Item)+" = "+gen(Value)
                else:
                    return 'call set_%s(%s, %s)'%(gen(Item), gen(Name), gen(Value))

            elif (ir.fn_defn, ir.void, Name, Attrs, Args, Excepts, Froms, Requires, Ensures, Body):
                return '''
                subroutine %s
                  %s
                  %s
                end subroutine %s
                ''' % (Name, gen(Args), gen(Body), Name)

            elif (ir.fn_defn, Typ, Name, Attrs, Args, Excepts, Froms, Requires, Ensures):
                return '''
                function %s
                  %s
                  %s
                end function %s
            ''' % (Typ, Name, gen(Args), gen(Body), Name)
            elif (ir.var_decl, Type, Name): declare_var(gen(Type), gen(Name))
            elif (ir.true):           return '.true.'
            elif (ir.false):          return '.false.'
            elif (Expr):
                return super(Fortran03CodeGenerator, self).generate(Expr, scope)

            else: raise Exception("match error")

# ----------------------------------------------------------------------
# C
# ----------------------------------------------------------------------
class CFile(SourceFile):
    """
    This class represents a C source file
    """
    def __init__(self, parent=None, relative_indent=0):
        #FIXME should be 0 see java comment
        super(CFile, self).__init__(parent, relative_indent)

    def __str__(self):
        """
        Perform the actual translation into a readable string,
        complete with indentation and newlines.
        """
        return self.dot_h() + self.dot_c()

    def dot_h(self):
        """
        Return a string of the header file declarations
        """
        s = sep_by(';\n', self._header)
        #if len(s) > 0:
        #    return ' '*self.relative_indent + s
        return s

    def dot_c(self):
        """
        Return a string of the c file declarations
        """
        return self._sep.join(self._defs)


class CCompoundStmt(CFile):
    """Represents a list of statements enclosed in braces {}"""
    def __init__(self, parent_scope):
        super(CCompoundStmt, self).__init__(parent_scope, relative_indent=2)

    def __str__(self):
        return (' {\n'
                + ' '*self.indent_level 
                + super(CCompoundStmt, self).__str__() +'\n'
                + ' '*(self.indent_level-2) + '}')
    
class ClikeCodeGenerator(GenericCodeGenerator):
    """
    C-like code generator
    """
    bin_op = {
        'log_or':  '||',
        'log_and': '&&',
        'eq':      '==',
        'ne':      '!=',
        'bit_or':  '|',
        'bit_and': '&',
        'bit_xor': '^',
        'lt':      '<',
        'gt':      '>',
        'lshift':  '<<',
        'rshift':  '>>',
        'plus':    '+',
        'minus':   '-',
        'times':   '*',
        'divide':  '/',
        'modulo':  '%',
        'rem':     'rem',
        'pow':     'pow'
        }

    un_op = { 
        'log_not': 'not',
        'bit_not': '~'
        }

    @generator
    @matcher(globals(), debug=False)
    def generate(self, node, scope=CFile()):
        # recursion
        def gen(node):
            return self.generate(node, scope)

        def new_def(s):
            #print 'new_def:', str(s)
            return scope.new_def(s)

        @accepts(str, tuple, str)
        def new_scope(prefix, body, suffix='\n'):
            '''used for things like if, while, ...'''
            comp_stmt = CCompoundStmt(scope)
            s = str(self.generate(body, comp_stmt))
            return new_def(prefix+s+suffix)

        def declare_var(typ, name):
            '''unless, of course, var were declared'''
            s = scope
            while not s.has_declaration_section():
                s = s.parent
            s.new_header_def(gen(typ)+' '+gen(name))
            return scope

        def gen_comma_sep(defs):
            return self.gen_in_scope(defs, Scope(relative_indent=1, separator=','))

        def gen_ws_sep(defs):
            return self.gen_in_scope(defs, Scope(relative_indent=0, separator=' '))

        def gen_dot_sep(defs):
            return self.gen_in_scope(defs, Scope(relative_indent=0, separator='.'))


        with match(node):
            if (ir.stmt, Expr):
                return new_def(gen(Expr)+';')

            elif (ir.fn_decl, Type, Name, Args):
                scope.new_header_def("%s %s(%s)"% (
                        gen(Type), gen(Name), gen_comma_sep(Args)))
                return scope

            elif (ir.fn_defn, Type, Name, Args, Body):
                return new_scope("%s %s(%s)"% (
                        gen(Type), gen(Name), gen_comma_sep(Args)), Body)

            elif (ir.return_, Expr):
                return "return %s" % gen(Expr)

            elif (ir.do_while, Condition, Body):
                return new_scope('do', Body, ' while (%s);'%gen(Condition))

            elif (ir.if_, Condition, Body):
                return new_scope('if (%s)'%gen(Condition), Body)

            elif (ir.arg, Attr, Mode, Type, Name): 
                return '%s %s'% (gen(Type), gen(Name))

            elif (ir.var_decl, Type, Name): 
                return declare_var(gen(Type), gen(Name))

            elif (ir.call, Name, Args): 
                return '%s(%s)' % (gen(Name), gen_comma_sep(Args))

            elif (ir.pointer_expr, Expr): return '&'+gen(Expr)
            elif (ir.pointer_type, Type): return '*'+str(gen(Type))
            elif (ir.deref):          return '*'
            elif (ir.log_not):        return '!'
            elif (ir.assignment):     return '='
            elif (ir.eq):             return '=='
            elif (ir.true):           return 'TRUE'
            elif (ir.false):          return 'FALSE'
            elif (Expr):
                return super(ClikeCodeGenerator, self).generate(Expr, scope)
            else: raise Exception("match error")

class CCodeGenerator(ClikeCodeGenerator):
    """
    C code generator
    """

    type_map = {
        'void':        "void",
        'bool':        "int",
        'character':   "char",
        'dcomplex':    "",
        'double':      "double",
        'fcomplex':    "",
        'float':       "float",
        'int':         "int",
        'long':        "long",
        'opaque':      "void*",
        'string':      "char*",
        'enum':        "enum",
        'struct':      "struct",
        'class':       "",
        'interface':   "",
        'package':     "",
        'symbol':      ""
        }


    @generator
    @matcher(globals(), debug=False)
    def generate(self, node, scope=CFile()):
        # recursion
        def gen(node):
            return self.generate(node, scope)

        def new_def(s):
            # print "new_def", s
            return scope.new_def(s)

        def pre_def(s):
            # print "pre_def", s
            return scope.pre_def(s)

        with match(node):
            if   (ir.struct, Name, _):         return gen(Name)
            elif (ir.primitive_type, Name, _): return gen(Name)

            elif (ir.get_struct_item, _, (ir.deref, StructName), (ir.struct_item, _, Item)):
                return gen(StructName)+'->'+gen(Item)

            elif (ir.set_struct_item, _, (ir.deref, StructName), (ir.struct_item, _, Item), Value):
                return gen(StructName)+'->'+gen(Item)+' = '+gen(Value)

            #FIXME: add a SIDL->C step that rewrites the SIDL struct accesses to use struct pointers

            elif (ir.get_struct_item, _, StructName, Item):
                return gen(StructName)+'.'+gen(Item)

            elif (ir.set_struct_item, _, StructName, Item, Value):
                return gen(StructName)+'.'+gen(Item)+' = '+gen(Value)

            elif (ir.struct, Name, _): return gen(Name)
            elif (ir.scoped_id, Names, Ext):
                return '_'.join([name for _, name in Names])

            elif (Expr):
                return super(CCodeGenerator, self).generate(Expr, scope)
            else: raise Exception("match error")


# ----------------------------------------------------------------------
# C++
# ----------------------------------------------------------------------
class CXXFile(CFile):
    """
    This class represents a C source file
    """
    def __init__(self):
        super(CXXFile, self).__init__()
    pass

class CXXCodeGenerator(CCodeGenerator):
    """
    C++ code generator
    """
    @matcher(globals(), debug=False)
    def get_type(self, node):
        """\return a string with the type of the IR node \c node."""
        return super(CXXCodeGenerator, self).get_type(node)

    @generator
    @matcher(globals(), debug=False)
    def generate(self, node, scope=CXXFile()):
        # recursion
        def gen(node):
            return self.generate(node, scope)

        def new_def(s):
            # print "new_def", s
            return scope.new_def(s)

        def pre_def(s):
            # print "pre_def", s
            return scope.pre_def(s)

        with match(node):
            if   (ir.get_struct_item, _, StructName, Item):
                return gen(StructName)+'.'+gen(Item)

            elif (ir.set_struct_item, _, StructName, Item, Value):
                return gen(StructName)+'.'+gen(Item)+' = '+gen(Value)

            elif (Expr):
                return super(CXXCodeGenerator, self).generate(Expr, scope)
            else: raise Exception("match error")

# ----------------------------------------------------------------------
# Java
# ----------------------------------------------------------------------
class JavaFile(SourceFile):
    """
    This class represents a Java source file
    """
    def __init__(self):
        #FIXME: file sould be 0 and there should be a class and package scope
        super(JavaFile, self).__init__(relative_indent=4) 

class JavaCodeGenerator(ClikeCodeGenerator):
    """
    Java code generator
    """
    type_map = {
        'void':        "void",
        'bool':        "boolean",
        'character':   "char",
        'dcomplex':    "",
        'double':      "double",
        'fcomplex':    "",
        'float':       "float",
        'int':         "int",
        'long':        "long",
        'opaque':      "",
        'string':      "String",
        'enum':        "enum",
        'struct':      "struct",
        'class':       "",
        'interface':   "",
        'package':     "",
        'symbol':      ""
        }

    @generator
    @matcher(globals(), debug=False)
    def generate(self, node, scope=JavaFile()):
        # recursion
        def gen(node):
            return self.generate(node, scope)

        def new_def(s):
            # print "new_def", s
            return scope.new_def(s)

        def pre_def(s):
            # print "pre_def", s
            return scope.pre_def(s)

        def get_function_scope():
            s = scope
            while s.parent != None:
                s = s.parent
            return s

        def deref((arg, struct, mode), structname):
            'dereference the holder object for inout and out arguments'
            if mode == ir.in_: 
                return gen(structname)
            else:
                s = get_function_scope()
                tmp = '_held_'+gen(structname)
                decl = '%s %s = %s.get();'%(
                    self.get_type(struct), tmp, gen(structname))
                if list(member(decl, s._header)) == []:
                    s.new_header_def(decl)
                return tmp

        with match(node):
            if   (ir.primitive_type, Type): return type_map[Type]
            elif (ir.struct, Package, Type, _): return gen(Package)+'.'+gen(Type)
            elif (ir.get_struct_item, Type, StructName, Item):
                return deref(Type, StructName)+'.'+gen(Item)

            elif (ir.set_struct_item, Type, StructName, Item, Value):
                return deref(Type, StructName)+'.'+gen(Item)+' = '+gen(Value)

            elif (ir.true):           return 'true'
            elif (ir.false):          return 'false'
            elif (Expr):
                return super(JavaCodeGenerator, self).generate(Expr, scope)
            else: raise Exception("match error")

# ----------------------------------------------------------------------
# Python
# ----------------------------------------------------------------------
class PythonFile(SourceFile):
    """
    This class represents a Python source file
    """
    def __init__(self, parent=None, relative_indent=4):
        super(PythonFile, self).__init__(parent, relative_indent)

    def __str__(self):
        """
        Perform the actual translation into a readable string,
        complete with indentation and newlines.
        """
        return ' '*self.indent_level+(
            '\n'+' '*self.indent_level).join(self._header+self._defs)+'\n'

    def break_line(self, string):
        """
        Break a string of Python code at max_line_length.
        """
        tokens = string.split()
        indent = ('\\\n'+re.match(r'^\s*', tokens[0]).group(0) +
                  ' '*self.relative_indent)
        lines = []
        while len(tokens) > 0:
            line = ""
            while (len(tokens) > 0 and 
                   len(line)+len(tokens[0]) < self._max_line_length):
                line += tokens.pop(0)+' '
            lines += [line]
        return indent.join(lines)


class PythonIndentedBlock(PythonFile):
    """Represents an indented block of statements"""
    def __init__(self, parent_scope):
        super(PythonIndentedBlock, self).__init__(
            parent_scope,
            indent_level=parent_scope.indent_level+4)
    def __str__(self):
        return (':\n'+
                super(PythonIndentedBlock, self).__str__())


class PythonCodeGenerator(GenericCodeGenerator):
    """
    Python code generator
    """
    bin_op = {
        'log_or':  'or',
        'log_and': 'and',
        'eq':      '==',
        'ne':      '<>',
        'bit_or':  '|',
        'bit_and': '&',
        'bit_xor': '^',
        'lt':      '<',
        'gt':      '>',
        'lshift':  '<<',
        'rshift':  '>>',
        'plus':    '+',
        'minus':   '-',
        'times':   '*',
        'divide':  '/',
        'modulo':  '%',
        'rem':     'rem',
        'pow':     'pow'
        }

    un_op = { 
        'log_not': '.not.',
        'bit_not': '~'
        }

    @generator
    @matcher(globals(), debug=False)
    def generate(self, node, scope=PythonFile()):
        # recursion
        def gen(node):
            return self.generate(node, scope)

        def new_def(s):
            # print "new_def", s
            return scope.new_def(s)

        def pre_def(s):
            # print "pre_def", s
            return scope.pre_def(s)

        def new_block(prefix, body, suffix='\n'):
            '''used for things like if, while, ...'''
            block = PythonIndentedBlock(scope)
            return new_def(prefix+
                           str(self.generate(body, block))+
                           suffix)

        with match(node):
            if (ir.fn_defn, Typ, Name, Attrs, Args, Excepts, Froms, Requires, Ensures):
                return '''
                def %s(%s):
                  %s
            ''' % (Name, gen(Args), gen(Body))
            elif ('return', Expr):
                return "return(%s)" % gen(Expr)

            elif (ir.get_struct_item, _, StructName, Item):
                return gen(StructName)+'.'+gen(Item)

            elif (ir.set_struct_item, _, StructName, Item, Value):
                return gen(StructName)+'.'+gen(Item)+' = '+gen(Value)

            elif (ir.do_while, Condition, Body):
                return new_block('while True', Body
                                 +[(ir.if_, (ir.prefix_expr(ir.log_not, Condition), 
                                             (ir.stmt, ir.break_)))])

            elif (ir.if_, Condition, Body):
                return new_block('if %s'%gen(Condition), Body)

            elif (ir.var_decl, Type, Name): return ''
            elif (ir.assignment):     return '='
            elif (ir.eq):             return '=='
            elif (ir.true):           return 'True'
            elif (ir.false):          return 'False'
            elif (Expr):
                return super(PythonCodeGenerator, self).generate(Expr, scope)
            else: raise Exception("match error")



# ----------------------------------------------------------------------
# SIDL
# ----------------------------------------------------------------------
class SIDLFile(Scope):
    """
    This class represents a SIDL source file
    """
    def __init__(self):
        super(SIDLFile, self).__init__()


class SIDLCodeGenerator(GenericCodeGenerator):
    """
    SIDL code generator
    """
    @generator
    @matcher(globals(), debug=False)
    def generate(self, node, scope=SIDLFile()):
        # recursion
        def gen(node):
            return self.generate(node, scope)

        def gen_(node):
            "like gen but with trailing ' ', if nonempty"
            if node == []: return ''
            return self.generate(node, scope)+' '

        def _gen(node):
            "like gen but with preceding ' ', if nonempty"
            if node == []: return ''
            return ' '+self.generate(node, scope)

        def _comma_gen(node):
            "like gen but with preceding ', ', if nonempty"
            if node == []: return ''
            return ', '+self.generate(node, scope)

        def new_def(s):
            if s == scope:
                import pdb; pdb.set_trace()
                raise Exception("Hey! No cycles, please.")
            if isinstance(s, list):
                import pdb; pdb.set_trace()
                raise Exception("Hey! No lists, neither.")
            #print "new_def", s
            if s <> '':
                scope.new_def(s)

        def pre_def(s):
            # print "pre_def", s
            return scope.pre_def(s)

        def gen_in_scope(defs, child_scope):
            r = self.generate(defs, child_scope)
            if (r <> ''):
                child_scope.new_def(r)
                #raise Exception("unexpected retval")
            return str(child_scope)

        def gen_scope(pre, defs, post):
            sep = '\n'+' '*scope.indent_level
            new_def(pre+sep+
                    gen_in_scope(defs, 
                                 Scope(scope, 4, 
                                       separator=';\n'))+';'+
                    sep+post)

        def gen_comma_sep(defs):
            return gen_in_scope(defs, Scope(relative_indent=1, separator=','))

        def gen_ws_sep(defs):
            return gen_in_scope(defs, Scope(relative_indent=0, separator=' '))

        def gen_dot_sep(defs):
            return gen_in_scope(defs, Scope(relative_indent=0, separator='.'))

        def tmap(f, l):
            return tuple(map(f, l))

        #import pdb; pdb.set_trace()

        with match(node):
            if (sidl.file, Requires, Imports, Packages):
                new_def(gen(Requires))
                new_def(gen(Imports))
                new_def(gen(Packages))
                return str(scope)

            elif (sidl.package, (Name), Version, Usertypes):
                gen_scope('package %s %s {' % (Name, gen(Version)),
                          Usertypes,
                          '}')

            elif (sidl.user_type, Attrs, Defn): 
                return gen_(Attrs)+gen(Defn)

            elif (sidl.class_, Name, Extends, Implements, Invariants, Methods):
                head = 'class '+gen(Name)
                if (Extends)    <> []: head += ' extends '+gen_ws_sep(Extends)
                if (Implements) <> []: head += ' implements '+gen_ws_sep(Implements)
                if (Invariants) <> []: head += ' invariants '+gen_ws_sep(Invariants)
                gen_scope(head+'{', Methods, '}')

            elif (sidl.interface, Name, Extends, Invariants, Methods):
                head = 'interface '+gen(Name)
                if (Extends)    <> []: head += ' extends '+gen_ws_sep(Extends)
                if (Invariants) <> []: head += ' invariants '+gen_ws_sep(Invariants)
                gen_scope(head+'{', Methods, '}')

            elif (sidl.method, Typ, Name, Attrs, Args, Excepts, Froms, Requires, Ensures):
                return (gen_ws_sep(Attrs)+
                        gen(Typ)+' '+gen(Name)+'('+gen_comma_sep(Args)+')'+
                        _gen(Excepts)+
                        _gen(Froms)+
                        _gen(Requires)+
                        _gen(Ensures))

            elif (sidl.arg, Attrs, Mode, Typ, Name):
                return gen_(Attrs) + '%s %s %s' % tmap(gen, (Mode, Typ, Name))

            elif (sidl.array, Typ, Dimension, Orientation):
                return ('array<%s%s%s>' % 
                        (gen(Typ), _comma_gen(Dimension), _comma_gen(Orientation)))

            elif (sidl.rarray, Typ, Dimension, Name, Extents):
                return ('rarray<%s%s> %s(%s)' %
                        (gen(Typ), _comma_gen(Dimension), gen(Name), gen_comma_sep(Extents)))

            elif (sidl.enum, Name, Enumerators):
                gen_scope('enum %s {' % gen(Name), gen_comma_sep(Enumerators), '}')

            elif (sidl.enumerator, Name):
                return gen(Name)

            elif (sidl.enumerator, Name, Value):
                return '%s = %s' % (gen(Name), gen(Value))

            elif (sidl.struct, Name, Items):
                gen_scope('struct %s {' % gen(Name), Items, '}')

            elif (sidl.scoped_id, A, B):
                return '%s%s' % (gen_dot_sep(A), gen(B))

            elif (sidl.type_attribute, Name):    return Name
            elif (sidl.version,     Version):    return 'version %s'%str(Version)
            elif (sidl.method_name, Name, []):   return gen(Name)
            elif (sidl.method_name, Name, Extension): return gen(Name)+' '+Extension
            elif (sidl.primitive_type, Name):    return Name
            elif (sidl.struct_item, Type, Name): return ' '.join((gen(Type), gen(Name)))
            elif (sidl.infix_expr, Op, A, B):    return ' '.join((gen(A), self.bin_op[Op], gen(B)))
            elif (sidl.prefix_expr, Op, A):      return ' '.join((self.un_op[Op], gen(A)))
            elif []: return ''
            elif A:
                if (isinstance(A, list)):
                    for defn in A:
                        new_def(gen(defn))
                else:
                    return str(A)
            else:
                raise Exception("match error")
        return ''

if __name__ == '__main__':
    try:
        print str(generate('C', ir.Plus(1, 2)))
        print [generate(lang, ir.Plus(1, 2)) for lang in ["C", "CXX", "F77", "F90", "F03", "Python", "Java"]] 
    except:
        # Invoke the post-mortem debugger
        import pdb, sys
        print sys.exc_info()
        pdb.post_mortem()

