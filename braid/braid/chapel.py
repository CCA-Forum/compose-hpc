#!/usr/bin/env python
# -*- python -*-
## @package chapel
#
# Babel functionality for the Chapel PGAS language
# http://chapel.cray.com/
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

import ir, sidl, re, os
from patmat import matcher, match, unify, expect, Variable
from codegen import (
    ClikeCodeGenerator, CCodeGenerator, 
    SourceFile, CFile, Scope, generator, accepts
)

def babel_object_type(package, name):
    """
    \return the SIDL node for the type of a Babel object 'name'
    \param name    the name of the object
    \param package the list of IDs making up the package
    """
    if isinstance(name, tuple):
        name = name[1]
    else: name = [name]
    return sidl.Scoped_id(package+name+['_object'], "")

def babel_exception_type():
    """
    \return the SIDL node for the Babel exception type
    """
    return babel_object_type(['sidl'], 'BaseInterface')

def ir_babel_object_type(package, name):
    """
    \return the IR node for the type of a Babel object 'name'
    \param name    the name of the object
    \param package the list of IDs making up the package
    """
    return ir.Struct(babel_object_type(package,name), [], '')

def ir_babel_exception_type():
    """
    \return the IR node for the Babel exception type
    """
    return ir_babel_object_type(['sidl'], 'BaseInterface')

@accepts(str, str)
def write_to(filename, string):
    """
    Create/Overwrite a file named \c filename with the contents of \c
    string.
    The file is written atomically.
    """
    tmp = '#'+filename+'#'
    f = open(tmp,'w')
    f.write(string)
    f.flush()
    os.fsync(f)
    f.close()
    os.rename(tmp, filename)


class Chapel:
    class ClassInfo:
        """
        Holder object for the code generation scopes and other data
        during the traversal of the SIDL tree.
        """
        def __init__(self, impl=None, stub=None, epv=None, ior=None):
            self.impl = impl
            self.stub = stub
            self.epv = epv
            self.ior = ior


    def __init__(self, filename, sidl_sexpr, create_makefile):
        """
        Create a new chapel code generator
        \param filename        full path to the SIDL file
        \param sidl_sexpr      s-expression of the SIDL data
        \param create_makefile if \c True, also generate a GNUmakefile
        """
        self.sidl_ast = sidl_sexpr
        self.sidl_file = filename
        self.create_makefile = create_makefile

    def generate_client(self):
        """
        Generate client code. Operates in two passes:
        \li create symbol table
        \li do the work
        """
        try:       
            sym = self.build_symbol_table(self.sidl_ast, SymbolTable())
            self.generate_client1(self.sidl_ast, None, sym) 
                  
        except:
            # Invoke the post-mortem debugger
            import pdb, sys
            print sys.exc_info()
            pdb.post_mortem()

    @matcher(globals(), debug=False)
    def generate_client1(self, node, data, symbol_table):
        def gen(node):         return self.generate_client1(node, data, symbol_table)
        def gen1(node, data1): return self.generate_client1(node, data1, symbol_table)

        if not symbol_table: 
            raise Exception()

        with match(node):
            if (sidl.file, Requires, Imports, UserTypes):
                gen(UserTypes)

            elif (sidl.user_type, Attrs, Cipse):
                gen(Cipse)

            elif (sidl.package, Name, Version, UserTypes, DocComment):
                self.generate_client1(UserTypes, data, symbol_table[Name])

            elif (sidl.class_, (Name), Extends, Implements, Invariants, Methods, DocComment):
                expect(data, None)
                # impl = ChapelFile()
                ci = self.ClassInfo(ChapelScope(), CFile(), EPV(Name, symbol_table), ior=CFile())
                ci.stub.genh(ir.Import(Name+'_IOR'))
                self.gen_default_methods(symbol_table, Name, ci)
                self.generate_ior(ci)
                gen1(Methods, ci)

                # IOR
                write_to(Name+'_IOR.h', ci.ior.dot_h(Name+'_IOR.h'))

                # Stub (in C)
                ci.stub.gen(ir.Import(Name+'_Stub'))
                # Stub Header
                write_to(Name+'_Stub.h', ci.stub.dot_h(Name+'_Stub.h'))
                # Stub C-file
                write_to(Name+'_Stub.c', ci.stub.dot_c())

                # Impl
                write_to(Name+'_Impl.chpl', str(ci.impl))

                # Makefile
                if self.create_makefile:
                    generate_makefile(self.sidl_file, Name)


            elif (sidl.method, Type, Name, Attrs, Args, Except, From, Requires, Ensures, DocComment):
                self.generate_client_method(symbol_table, node, data)               

            elif A:
                if (isinstance(A, list)):
                    for defn in A:
                        gen(defn)
                else:
                    raise Exception("NOT HANDLED:"+repr(A))
            else:
                raise Exception("match error")
        return data

    @matcher(globals(), debug=False)
    def build_symbol_table(self, node, symbol_table):
        """
        Build a hierarchical \c SymbolTable() for \c node.

        For the time being, we store the fully scoped name
        (= \c [package,subpackage,classname] ) for each class 
        in the symbol table.
        """

        def gen(node):
            return self.build_symbol_table(node, symbol_table)

        with match(node):
            if (sidl.file, Requires, Imports, UserTypes): 
                gen(UserTypes)

            elif (sidl.user_type, Attrs, Cipse): 
                gen(Cipse)

            elif (sidl.package, Name, Version, UserTypes, DocComment):
                symbol_table[Name] = SymbolTable(symbol_table, 
                                                 symbol_table.prefix+[Name])
                self.build_symbol_table(UserTypes, symbol_table[Name])

            elif (sidl.class_, Name, Extends, Implements, Invariants, Methods, DocComment):
                symbol_table[Name] = \
                    ( sidl.class_, (sidl.scoped_id, symbol_table.prefix+[Name], []),
                      Extends, Implements, Invariants, Methods )

            elif (sidl.struct, (sidl.scoped_id, Names, Ext), Items, DocComment):
                symbol_table[Name] = \
                    ( sidl.struct, 
                      (sidl.scoped_id, symbol_table.prefix+Names, []), 
                      Items )

            elif A:
                if (isinstance(A, list)):
                    for defn in A:
                        gen(defn)
                else:
                    raise Exception("NOT HANDLED:"+repr(A))

            else:
                raise Exception("match error")

        return symbol_table

    def gen_default_methods(self, symbol_table, name, data):
        data.epv.add_method(sidl.Method(
                sidl.void,
                sidl.Method_name("_cast", ''), [],
                [sidl.Arg([], sidl.in_, babel_object_type(symbol_table.prefix, name), 
                          'self'), 
                 sidl.Arg([], sidl.in_, sidl.Primitive_type(sidl.string), 'name'), 
                 sidl.Arg([], sidl.in_, babel_exception_type(), 'ex')],
                [], [], [], [], 'Cast'))

    @matcher(globals(), debug=False)
    def generate_client_method(self, symbol_table, method, ci):
        """
        Generate client code for a method interface.
        \param method        s-expression of the method's SIDL declaration
        \param symbol_table  the symbol table of the SIDL file
        \param ci            a ClassInfo object
        """
        ci.epv.add_method(method)
        # output _extern declaration
        ci.impl.new_def('_extern '+ chpl_gen(method))
        # output the stub definition
        stub = self.generate_stub(symbol_table, method, ci)
        c_gen(stub, ci.stub)


    def generate_stub(self, symbol_table,
                      (Method, Type, (_,  Name, _Attr), Attrs, Args, 
                       Except, From, Requires, Ensures, DocComment), ci):

        def argname((_arg, _attr, _mode, _type, Id)):
            return Id
        def low(sidl_term):
            return lower_ir(symbol_table, sidl_term)

        #return method
        expect(Method, sidl.method)
        decl_args = babel_args(Args, symbol_table, ci.epv.name) 
        call_args = ['self'] + map(argname, Args) + ['ex'] 
        epv_type = ci.epv.get_sexpr()
        obj_type = ci.obj
        Body = [ir.Stmt
                (ir.Return
                 (ir.Call
                  (ir.Get_struct_item
                   (epv_type,
                    ir.Get_struct_item
                    (obj_type, 
                     ir.Deref('self'),
                     ir.Struct_item(ir.Primitive_type(ir.void), 'f_'+Name)),
                    ir.Struct_item(ir.Primitive_type(ir.void), 'd_data')),
                   call_args)))]
        return [ir.Fn_decl(low(Type), Name, decl_args, DocComment),
                ir.Fn_defn(low(Type), Name, decl_args, Body, DocComment)]

    def generate_ior(self, ci):
        epv = ci.epv.get_sexpr()
        cstats = \
            ir.Struct(ir.Scoped_id(ci.epv.symbol_table.prefix+[ci.epv.name,'_cstats'], ''), 
                      [ir.Struct_item(ir.Typedef_type("sidl_bool"), "use_hooks")], 
                       'The controls and statistics structure')
        ci.obj = \
            ir.Struct(ir.Scoped_id(ci.epv.symbol_table.prefix+[ci.epv.name,'_object'], ''), 
                      [ir.Struct_item(ir_babel_object_type(['sidl'], 'BaseClass'), 
                                      "d_sidl_baseclass"),
                       ir.Struct_item(ir.Pointer_type(epv), "d_epv"), 
                       ir.Struct_item(cstats, "d_cstats"),
                       ir.Struct_item(ir.Pointer_type(ir.Primitive_type(ir.void)), "d_data")
                       ],
                       'The class object structure')
            
        ci.ior.genh(ir.Import('sidl'))
        ci.ior.genh(ir.Import('sidl_BaseInterface_IOR'))
        ci.ior.gen(ir.Type_decl(cstats))
        ci.ior.gen(ir.Type_decl(ci.obj))
        ci.ior.gen(ir.Type_decl(epv))

@matcher(globals(), debug=False)
def lower_ir(symbol_table, sidl_term):
    """
    lower SIDL into IR
    """
    def low(sidl_term):
        return lower_ir(symbol_table, sidl_term)

    def low_t(sidl_term):
        return lower_type_ir(symbol_table, sidl_term)

    with match(sidl_term):
        if (sidl.struct, Name, Items):
            return ir.Pointer_expr(ir.Struct(low_t(Name), Items, ''))

        elif (sidl.arg, Attrs, Mode, Typ, Name): 
            return ir.Arg(Attrs, Mode, low_t(Typ), Name)

        elif (sidl.void):                 return ir.Primitive_type(ir.void)
        elif (sidl.primitive_type, Type): return low_t(sidl_term)

        elif (Terms):        
            if (isinstance(Terms, list)):
                return map(low, Terms)
        else:
            raise Exception("Not implemented")

@matcher(globals(), debug=False)
def lower_type_ir(symbol_table, sidl_type):
    """
    lower SIDL types into IR
    """
    with match(sidl_type):
        if (sidl.scoped_id, Names, Ext):
            return lower_type_ir(symbol_table, lookup_type(symbol_table, Names))
        elif (sidl.void):                        return ir.Primitive_type(ir.void)
        elif (sidl.primitive_type, sidl.opaque): return ir.Pointer_type(ir.Primitive_type(ir.void))
        elif (sidl.primitive_type, sidl.string): return ir.const_str
        elif (sidl.primitive_type, Type):        return ir.Primitive_type(Type)
        elif (sidl.class_, Name, _, _, _, _):    
            return ir_babel_object_type([], Name)
        else:
            raise Exception("Not implemented")
 
def lookup_type(symbol_table, scopes):
    """
    perform a symbol lookup of a scoped identifier
    """
    n = len(scopes)
    # go up (and down again) in the hierarchy
    # FIXME: Is this the expected bahavior for nested packages?
    sym = symbol_table[scopes[0]]
    while not sym: # up until we find something
        symbol_table = symbol_table.parent()
        sym = symbol_table[scopes[0]]
        
    for i in range(1, n-1): # down again to resolve it
        sym = sym[scopes[i]]
    
    if not sym:
        raise Exception("Symbol lookup error: "+repr(key))

    #print "successful lookup(", symbol_table, ",", scopes, ") =", sym
    return sym

class SymbolTable:
    """
    Hierarchical symbol table for SIDL identifiers.
    \arg prefix  parent package. A list of identifiers 
                 just as they would appear in a \c Scoped_id()
    """
    def __init__(self, parent=None, prefix=[]):
        #print "new scope", self, 'parent =', parent
        self._parent = parent
        self._symbol = {}
        self.prefix = prefix

    def parent(self):
        if self._parent: 
            return self._parent
        else: 
            raise Exception("Symbol lookup error: no parent scope")

    @matcher(globals())
    def __getitem__(self, key):
        #print self, key, '?'
        try:
            return self._symbol[key]
        except KeyError:
            return None

    @matcher(globals())
    def __setitem__(self, key, value):
        #print self, key, '='#, value
        self._symbol[key] = value

class EPV:
    """
    Babel entry point vector for virtual method calls.
    """
    def __init__(self, name, symbol_table):
        self.methods = []
        self.name = name
        self.symbol_table = symbol_table

    def add_method(self, method):
        """
        add another (SIDL) method to the vector
        """
        def to_fn_decl((_sidl_method, Type, 
                        (Method_name, Name, Extension), 
                        Attrs, Args, Except, From, Requires, Ensures, DocComment)):
            typ = lower_ir(self.symbol_table, Type)
            name = 'f_'+Name
            args = babel_args(Args, self.symbol_table, self.name)
            return ir.Fn_decl(typ, name, args, DocComment)

        self.methods.append(to_fn_decl(method))
        return self

    def get_sexpr(self):
        """
        return an s-expression of the EPV declaration
        """
        def get_type_name((fn_decl, Type, Name, Args, DocComment)):
            return ir.Pointer_type((fn_decl, Type, Name, Args, DocComment)), Name

        name = ir.Scoped_id(self.symbol_table.prefix+[self.name,'_epv'], '')
        return ir.Struct(name,
            [ir.Struct_item(itype, iname) 
             for itype, iname in map(get_type_name, self.methods)], 
                         'Entry Point Vector (EPV)')

def babel_args(args, symbol_table, class_name):
    """
    \return [self]+args+[ex]
    """
    arg_self = ir.Arg([], sidl.in_, ir_babel_object_type(symbol_table.prefix, 
                                                           class_name), 'self')
    arg_ex = ir.Arg([], sidl.in_, ir_babel_exception_type(), 'ex')
    return [arg_self]+lower_ir(symbol_table, args)+[arg_ex]

def arg_ex():
    """
    default SIDL exception argument
    """
    return 


def chpl_gen(ir):
    return str(ChapelCodeGenerator().generate(ir, ChapelScope()))

def c_gen(ir, scope=CFile()):
    return CCodeGenerator().generate(ir, scope)

class ChapelFile(SourceFile):
    def __init__(self, parent=None, relative_indent=0):
        super(ChapelFile, self).__init__(
            parent, relative_indent, separator=';\n')

    def gen(self, ir):
        """
        Invoke the Chapel code generator on \c ir and append the result to
        this ChapelFile object.
        """
        ChapelCodeGenerator().generate(ir, self)


class ChapelScope(ChapelFile):
    def __init__(self, parent=None, relative_indent=2):
        super(ChapelScope, self).__init__(parent, relative_indent)


class ChapelCodeGenerator(ClikeCodeGenerator):
    type_map = { 
        'void':      "void",
        'bool':      "logical",
        'character': "character",
        'dcomplex':  "double complex",
        'double':    "double precision",
        'fcomplex':  "complex",
        'float':     "real",
        'int':       "int",
        'long':      "int",
        'opaque':    "int",
        'string':    "character",
        'enum':      "integer",
        'struct':    "integer",
        'class':     "integer",
        'interface': "integer",
        'package':   "void",
        'symbol':    "integer"
        }

    @generator
    @matcher(globals(), debug=False)
    def generate(self, node, scope=ChapelFile()):
        def gen(node):
            return self.generate(node, scope)

        def new_def(s):
            return scope.new_def(s)

        def gen_comma_sep(defs):
            return self.gen_in_scope(defs, Scope(relative_indent=1, separator=','))

        def gen_ws_sep(defs):
            return self.gen_in_scope(defs, Scope(relative_indent=0, separator=' '))

        def gen_dot_sep(defs):
            return self.gen_in_scope(defs, Scope(relative_indent=0, separator='.'))

        def tmap(f, l):
            return tuple(map(f, l))

        def gen_comment(doc_comment):
            if doc_comment == '': 
                return ''
            sep = '\n'+' '*scope.indent_level
            return (sep+' * ').join(['/**']+
                                   re.split('\n\s*', doc_comment)
                                   )+sep+' */'+sep

        with match(node):
            if (sidl.method, 'void', Name, Attrs, Args, Except, From, Requires, Ensures, DocComment):
                new_def('%sdef %s(%s)'%(gen_comment(DocComment), gen(Name), gen_comma_sep(Args)))

            elif (sidl.method, Type, Name, Attrs, Args, Except, From, Requires, Ensures, DocComment):
                new_def('%sdef %s(%s): %s'%(gen_comment(DocComment),
                                            gen(Name), gen_comma_sep(Args), gen(Type)))

            elif (sidl.arg, Attrs, Mode, Type, Name):
                return '%s: %s'%(gen(Name), gen(Type))

            elif (sidl.class_, (Name), Extends, Implements, Invariants, Methods, Package, DocComment):
                return gen_comment(DocComment)+'class '+Name

            elif (sidl.primitive_type, Type):       return self.type_map[Type]
            elif (sidl.custom_attribute, Id):       return gen(Id)
            elif (sidl.method_name, Id, []):        return gen(Id)
            elif (sidl.method_name, Id, Extension): return gen(Id)
            elif (sidl.scoped_id, A, B):
                return '%s%s' % (gen_dot_sep(A), gen(B))

            elif (Expr):
                return super(ChapelCodeGenerator, self).generate(Expr, scope)

            else:
                raise Exception('match error')
        return scope

def generate_makefile(sidl_file, classnames):
    """
    FIXME: make this a file copy from $prefix/share
           make this work for more than one class
    """
    write_to('babel.make', """
IORHDRS = {file}_IOR.h
STUBHDRS = {file}.h
STUBSRCS = {file}_Stub.c
""".format(file=classnames))

    write_to('GNUmakefile', r"""
# Generic Chapel Babel wrapper GNU Makefile
# $Id$
# 
# Copyright (c) 2008, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the Components Team <components@llnl.gov>
# UCRL-CODE-2002-054
# All rights reserved.
# 
# This file is part of Babel. For more information, see
# http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
# for Our Notice and the LICENSE file for the GNU Lesser General Public
# License.
# 
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License (as published by
# the Free Software Foundation) version 2.1 dated February 1999.
# 
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
# conditions of the GNU Lesser General Public License for more details.
# 
# You should have recieved a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
# This Makefile uses GNU make extensions, so it may not work with
# other implementations of make.

include babel.make
# please name the server library here
LIBNAME=impl
# please name the SIDL file here
SIDLFILE="""+sidl_file+r"""
# extra include/compile flags 
EXTRAFLAGS=
# extra libraries that the implementation needs to link against
EXTRALIBS=
# library version number
VERSION=0.1.1
# PREFIX specifies the top of the installation directory
PREFIX=/usr/local
# the default installation installs the .la and .scl (if any) into the
# LIBDIR
LIBDIR=$(PREFIX)/lib
# the default installation installs the stub header and IOR header files
# in INCLDIR
INCLDIR=$(PREFIX)/include


# most of the rest of the file should not require editing

ifeq ($(IMPLSRCS),)
  SCLFILE=
  BABELFLAG=--client=Chapel
  MODFLAG=
else
  SCLFILE=lib$(LIBNAME).scl
  BABELFLAG=--server=Chapel
  MODFLAG=-module
endif

all : lib$(LIBNAME).la $(SCLFILE)

CC=`babel-config --query-var=CC`
INCLUDES=`babel-config --includes` -I.
CFLAGS=`babel-config --flags-c`
LIBS=`babel-config --libs-c-client`

STUBOBJS=$(STUBSRCS:.c=.lo)
IOROBJS=$(IORSRCS:.c=.lo)
SKELOBJS=$(SKELSRCS:.c=.lo)
IMPLOBJS=$(IMPLSRCS:.c=.lo)

PUREBABELGEN=$(IORHDRS) $(IORSRCS) $(STUBSRCS) $(STUBHDRS) $(SKELSRCS)
BABELGEN=$(IMPLHDRS) $(IMPLSRCS)

$(IMPLOBJS) : $(STUBHDRS) $(IORHDRS) $(IMPLHDRS)

lib$(LIBNAME).la : $(STUBOBJS) $(IOROBJS) $(IMPLOBJS) $(SKELOBJS)
	babel-libtool --mode=link --tag=CC $(CC) -o lib$(LIBNAME).la \
	  -rpath $(LIBDIR) -release $(VERSION) \
	  -no-undefined $(MODFLAG) \
	  $(CFLAGS) $(EXTRAFLAGS) $^ $(LIBS) \
	  $(EXTRALIBS)

$(PUREBABELGEN) $(BABELGEN) : babel-stamp
	@if test -f $@; then \
	    touch $@; \
	else \
	    rm -f babel-stamp ; \
	    $(MAKE) babel-stamp; \
	fi

babel-stamp: $(SIDLFILE)
	@rm -f babel-temp
	@touch babel-temp
	braid $(BABELFLAG) $(SIDLFILE) 
	@mv -f babel-temp $@

lib$(LIBNAME).scl : $(IORSRCS)
ifeq ($(IORSRCS),)
	echo "lib$(LIBNAME).scl is not needed for client-side C bindings."
else
	-rm -f $@
	echo '<?xml version="1.0" ?>' > $@
	echo '<scl>' >> $@	
	if test `uname` = "Darwin"; then scope="global"; else scope="local"; \
	   fi ; \
          echo '  <library uri="'`pwd`/lib$(LIBNAME).la'" scope="'"$$scope"'" resolution="lazy" >' >> $@
	grep __set_epv $^ /dev/null | awk 'BEGIN {FS=":"} { print $$1}' | sort -u | sed -e 's/_IOR.c//g' -e 's/_/./g' | awk ' { printf "    <class name=\"%s\" desc=\"ior/impl\" />\n", $$1 }' >>$@
	echo "  </library>" >>$@
	echo "</scl>" >>$@
endif

.SUFFIXES: .lo

.c.lo:
	babel-libtool --mode=compile --tag=CC $(CC) $(INCLUDES) $(CFLAGS) $(EXTRAFLAGS) -c -o $@ $<

clean : 
	-rm -f $(PUREBABELGEN) babel-temp babel-stamp *.o *.lo

realclean : clean
	-rm -f lib$(LIBNAME).la lib$(LIBNAME).scl
	-rm -rf .libs

install : install-libs install-headers install-scl


install-libs : lib$(LIBNAME).la
	-mkdir -p $(LIBDIR)
	babel-libtool --mode=install install -c lib$(LIBNAME).la \
	  $(LIBDIR)/lib$(LIBNAME).la

install-scl : $(SCLFILE)
ifneq ($(IORSRCS),)
	-rm -f $(LIBDIR)/lib$(LIBNAME).scl
	-mkdir -p $(LIBDIR)
	echo '<?xml version="1.0" ?>' > $(LIBDIR)/lib$(LIBNAME).scl
	echo '<scl>' >> $(LIBDIR)/lib$(LIBNAME).scl	
	if test `uname` = "Darwin"; then scope="global"; else scope="local"; \
	   fi ; \
          echo '  <library uri="'$(LIBDIR)/lib$(LIBNAME).la'" scope="'"$$scope"'" resolution="lazy" >' >> $(LIBDIR)/lib$(LIBNAME).scl
	grep __set_epv $^ /dev/null | awk 'BEGIN {FS=":"} { print $$1}' | sort -u | sed -e 's/_IOR.c//g' -e 's/_/./g' | awk ' { printf "    <class name=\"%s\" desc=\"ior/impl\" />\n", $$1 }' >>$(LIBDIR)/lib$(LIBNAME).scl
	echo "  </library>" >>$(LIBDIR)/lib$(LIBNAME).scl
	echo "</scl>" >>$(LIBDIR)/lib$(LIBNAME).scl
endif

install-headers : $(IORHDRS) $(STUBHDRS)
	-mkdir -p $(INCLDIR)
	for i in $^ ; do \
	  babel-libtool --mode=install cp $$i $(INCLDIR)/$$i ; \
	done

.PHONY: all clean realclean install install-libs install-headers install-scl
""")
