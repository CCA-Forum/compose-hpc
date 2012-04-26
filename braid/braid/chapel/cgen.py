#!/usr/bin/env python
# -*- python -*-
## @package chapel.cgen
#
# BRAID code generator implementation for Chapel
#
# Please report bugs to <adrian@llnl.gov>.
#
# \authors <pre>
#
# Copyright (c) 2011, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Written by Adrian Prantl <adrian@llnl.gov>.
#
# Contributors/Acknowledgements:
#
# Summer interns at LLNL:
# * 2010, 2011 Shams Imam <shams@rice.edu>
#   contributed argument conversions, r-array handling, exception handling,
#   distributed arrays, the patches to the Chapel compiler, ...
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
import ir, sidl, re
import ior
from patmat import *
from utils import *
from codegen import (
    ClikeCodeGenerator, CCodeGenerator, c_gen,
    SourceFile, CFile, CCompoundStmt, Scope, generator, accepts,
    sep_by
)
import conversions as conv

def drop(lst):
    """
    If \c lst is \c [] return \c None, else return \c lst[0] .
    """
    if lst: return lst[0]
    else:   return None

def epv_qname(scoped_id, sep='_'):
    """
    get rid of the __epv suffix when constructing a qual_id
    """
    _, prefix, name, _ = scoped_id
    return sep.join(prefix+[name[:-5]])

# FIXME: this is a non-reentrant hack to create each stub only once
stubs_generated = set()

def incoming((arg, attrs, mode, typ, name)):
    return mode <> sidl.out

def outgoing((arg, attrs, mode, typ, name)):
    return mode <> sidl.in_

def ir_arg_to_chpl((arg, attrs, mode, typ, name)):
    chpl_type = conv.ir_type_to_chpl(typ)
#    if mode == ir.in_ and typ[0] == ir.pointer_type and typ[1][0] == ir.struct and typ <> chpl_type:
#        # IN structs are passed by value in Chapel
#        return arg, attrs, mode, chpl_type[1], name
    return arg, attrs, mode, chpl_type, name

def deref(mode, typ, name):
    if typ[0] == ir.pointer_type and typ[1][0] == ir.struct:
        return name+'->'
    if typ[0] == ir.pointer_type and typ[1][0] == ir.pointer_type and typ[1][0][0] == ir.struct: # _ex
        return name+'->'
    #elif typ[0] ==  ir.struct:
    #    return name+'->'
    elif mode == sidl.in_:
        return name 
    else: return '(*%s)'%name

def deref_retval(typ, name):
    if typ[0] == ir.struct:
        return '(*%s)'%name
    else: return name

def strip(typ):
    if typ[0] == ir.pointer_type and typ[1][0] == ir.struct:
        return ir.struct
    if typ[0] == ir.typedef_type and typ[1] == 'sidl_bool':
        return ior.bool
    # strip unnecessary details from aggregate types
    if (typ[0] == ir.enum or
        #typ[0] == sidl.array or
        typ[0] == sidl.rarray or
        typ[0] == ir.pointer_type or
        typ[0] == ir.struct):
        return typ[0]
    return typ


def need_return_arg(typ):
    """
    \return True iff the type \c typ cannot be passed as a return
    value in Chapel.
    """
    return typ[0] == ir.struct

def generate_method_stub(scope, (_call, VCallExpr, CallArgs), scoped_id):
    """
    Generate the stub for a specific method in C (cStub).

    \return   if the methods needs an extra inout argument for the
              return value, this function returns its type, otherwise
              it returns \c None.
    """

    def obj_by_value((arg, attrs, mode, typ, name)):
        if typ[0] == ir.pointer_type and typ[1][0] == ir.struct and name == 'self':
            return (arg, attrs, ir.in_, typ, name)
        elif typ[0] == ir.pointer_type and typ[1][0] == ir.struct and mode == ir.in_:
            return (arg, attrs, ir.inout, typ, name)
        else:
            return (arg, attrs, mode, typ, name)

    if VCallExpr[0] == ir.deref:
        _, (_, _ , _, (_, (_, impl_decl), _)) = VCallExpr
    else:
        _, _ , _, (_, (_, impl_decl), _) = VCallExpr

    (_, Attrs, Type, Name, Args, DocComment) = impl_decl
    sname = '_'.join([epv_qname(scoped_id), Name, 'stub'])

    # convert arguments to/from IOR using proxy variables
    pre_call = []
    post_call = []
    retval_arg = []
    opt = scope.cstub.optional

    # IN
    map(lambda (arg, attr, mode, typ, name):
          conv.codegen((('chpl', strip(typ)), deref(mode, typ, name)), strip(typ),
                       pre_call, scope, '_ior_'+name, typ),
        filter(incoming, Args))

    # OUT
    map(lambda (arg, attr, mode, typ, name):
          conv.codegen((strip(typ), '_ior_'+name), ('chpl', strip(typ)),
                       post_call, scope, '(*%s)'%name, typ),
        filter(outgoing, Args))

    cstub_decl_args = map(ir_arg_to_chpl, Args)

    # RETURN value type conversion -- treated like an out argument
    rarg = ir.Arg([], ir.out, Type, '_retval')
    conv.codegen((strip(Type), '_ior__retval'), ('chpl', strip(Type)), 
                 post_call, scope, deref_retval(Type, '_retval'), Type)
    crarg = ir_arg_to_chpl(rarg)
    _,_,_,chpltype,_ = crarg
    retval_type = chpltype
    if need_return_arg(Type):
        # pass the return value as an extra out argument
        chpltype = ir.pt_void
        cstub_decl_args.append(crarg)

    # Proxy declarations / revised names of call arguments
    call_args = []
    decls = []
    for (_,attrs,mode,chpl_t,name), (_,_,_,c_t,_) in (
        zip(cstub_decl_args+[crarg], Args+[rarg])):
        if chpl_t <> c_t:
            need_deref = False
            if c_t[0] == ir.pointer_type and c_t[1][0] == ir.struct:
                c_t = c_t[1]
                need_deref = True

            # FIXME see comment in chpl_to_ior
            name = '_ior_'+name
            decls.append(ir.Stmt(ir.Var_decl(c_t, name)))
            if mode <> sidl.in_ or need_deref:
                name = ir.Pointer_expr(name)

        if name == 'self' and member_chk(ir.pure, attrs): # part of the hack for self dereferencing
            upcast = ('({0}*)(((struct sidl_BaseInterface__object*)self)->d_object)'
                      .format(c_gen(c_t[1])))
            call_args.append(upcast)
        else:
            call_args.append(name)

    # get rid of retval in call args
    retval_name = call_args[-1] if isinstance(call_args[-1], str) else call_args[-1][1]
    call_args = call_args[:-1]

    cstub_decl = ir.Fn_decl([], chpltype, sname, cstub_decl_args, DocComment)

    if Type == ir.pt_void:
        body = [ir.Stmt((ir.call, VCallExpr, call_args))]
    else:
        if chpltype <> ir.pt_void:
            pre_call.append(ir.Stmt(ir.Var_decl(retval_type, '_retval')))

        body = [ir.Stmt(ir.Assignment(retval_name, (ir.call, VCallExpr, call_args)))]

        if chpltype <> ir.pt_void:
            post_call.append(ir.Stmt(ir.Return('_retval')))

    # Generate the C code into the scope's associated cStub
    if sname not in stubs_generated:
        stubs_generated.add(sname)
        c_gen([cstub_decl,
           ir.Fn_defn([], chpltype, sname, cstub_decl_args,
                      decls+pre_call+body+post_call, DocComment)], scope.cstub)

    # Chapel extern declaration
    chplstub_decl = ir.Fn_decl([], chpltype, sname, map(obj_by_value, cstub_decl_args), DocComment)
    scope.new_header_def('extern '+chpl_gen(chplstub_decl)+';')

    return drop(retval_arg)



class ChapelFile(SourceFile):
    """
    A BRAID-style code generator output file manager for the Chapel language.

    * Chapel files also have a cstub which is used to output code that
      can not otherwise be expressed in Chapel.

    * The main_area member denotes the space that defaults to the
      module's main() function.
    """

    @accepts(object, str, object, int, str)
    def __init__(self, name="", parent=None, relative_indent=0, separator='\n'):
        super(ChapelFile, self).__init__(
            name, parent, relative_indent, separator=separator)
        if parent:
            self.cstub = parent.cstub
            self.main_area = parent.main_area
        else:
            self.cstub = CFile()
            # This is for definitions that are generated in multiple
            # locations but should be written out only once.
            self.cstub.optional = set()
            # Tricky circular initialization
            self.main_area = None
            main_area = ChapelScope(self, relative_indent=0)
            self.main_area = main_area

    def __str__(self):
        """
        Perform the actual translation into a readable string,
        complete with indentation and newlines.
        """
        if self.parent:
            main = ''
        else: # output main only at the toplevel
            main = str(self.main_area)

        h_indent = ''
        d_indent = ''
        if len(self._header)   > 0: h_indent=self._sep
        if len(self._defs)     > 0: d_indent=self._sep

        return ''.join([
            h_indent,
            sep_by(';'+self._sep, self._header),
            d_indent,
            self._sep.join(self._defs),
            main
            ])

    def get_decls(self):
        h_indent = ''
        if len(self._header) > 0:
            h_indent=self._sep
        return ''.join([h_indent, sep_by(';'+self._sep, self._header)])

    def get_defs(self):
        d_indent = ''
        if len(self._defs) > 0:
            d_indent=self._sep
        return ''.join([d_indent, sep_by(self._sep, self._defs)])


    def gen(self, ir):
        """
        Invoke the Chapel code generator on \c ir and append the result to
        this ChapelFile object.
        """
        ChapelCodeGenerator().generate(ir, self)

    def write(self):
        """
        Atomically write the ChapelFile and its cStub to disk, using the
        basename provided in the constructor.
        Empty files will not be created.
        """
        if self._defs or self._header:
            write_to(self._name+'.chpl', str(self))
        self.cstub.write()

    def sub_scope(self, relative_indent, separator):
        return ChapelLine(self, relative_indent, separator)

class ChapelScope(ChapelFile):
    """
    A Chapel scope, ie., a block of statements enclosed by curly braces.
    """
    @accepts(object, object, str)
    def __init__(self, parent=None, relative_indent=4):
        super(ChapelScope, self).__init__(parent=parent, 
                                          relative_indent=relative_indent)

    def __str__(self):
        if self.main_area == None:
            self._sep = ';\n'
            terminator = ';\n';
        else:
            terminator = ''

        return '%s%s'%(self._sep.join(self._header+self._defs), terminator)

class ChapelLine(ChapelFile):
    """
    A single line of Chapel code, such as a statement.
    """
    @accepts(object, object, int, str)
    def __init__(self, parent=None, relative_indent=0, separator='\n'):
        super(ChapelLine, self).__init__(parent=parent, 
                                         relative_indent=relative_indent, 
                                         separator=separator)

    def __str__(self):
        return self._sep.join(self._header+self._defs)


def chpl_gen(ir, scope=None):
    """
    Generate Chapel code with the optional scope argument
    \return a string
    """
    if scope == None:
        scope = ChapelScope()
    return str(ChapelCodeGenerator().generate(ir, scope))

def gen_doc_comment(doc_comment, scope):
    if doc_comment == '':
        return ''
    sep = '\n'+' '*scope.indent_level
    return (sep+' * ').join(['/**']+
                           re.split('\n\s*', doc_comment)
                           )+sep+' */'+sep

sidl_array_regex = re.compile('^sidl(_(\w+))__array$')

class ChapelCodeGenerator(ClikeCodeGenerator):
    """
    A BRAID-style code generator for Chapel.
    """
    type_map = {
        'void':      "void",
        'bool':      "bool",
        'char':      "string",
        'dcomplex':  "complex(128)",
        'double':    "real(64)",
        'fcomplex':  "complex(64)",
        'float':     "real(32)",
        'int':       "int(32)",
        'long':      "int(64)",
        'opaque':    "int(64)",
        'string':    "string"
        }

    def is_sidl_array(self, struct_name):
        m = sidl_array_regex.match(struct_name)
        if m:
            t = m.group(2)
            try:
                return self.type_map[t]
            except:
                return 'opaque'
        return None


    @generator
    @matcher(globals(), debug=False)
    def generate(self, node, scope=ChapelFile()):
        """
        This code generator is a bit unusual in that it accepts a
        hybrid of \c sidl and \c ir nodes.
        """
        def gen(node):
            return self.generate(node, scope)

        def new_def(s):
            return scope.new_def(s)

        def new_header_def(s):
            return scope.new_header_def(s)

        @accepts(str, list, str)
        def new_scope(prefix, body, suffix='\n'):
            '''used for things like if, while, ...'''
            comp_stmt = ChapelFile(parent=scope, relative_indent=4)
            s = str(self.generate(body, comp_stmt))
            return new_def(scope._sep.join(['',prefix+s,suffix]))

        @accepts(str, str, str)
        def new_scope1(prefix, body, suffix):
            '''used for things like enumerator'''
            return scope.new_header_def(''.join([prefix,body,suffix])+';')

        def gen_comma_sep(defs):
            return self.gen_in_scope(defs, scope.sub_scope(relative_indent=1, separator=','))

        def gen_semicolon_sep(defs):
            return self.gen_in_scope(defs, scope.sub_scope(relative_indent=2, separator=';\n'))+';'

        def gen_ws_sep(defs):
            return self.gen_in_scope(defs, scope.sub_scope(relative_indent=0, separator=' '))

        def gen_dot_sep(defs):
            return self.gen_in_scope(defs, scope.sub_scope(relative_indent=0, separator='.'))

        def tmap(f, l):
            return tuple(map(f, l))

        def gen_comment(DocComment):
            return gen_doc_comment(DocComment, scope)

        def gen_attrs(attrs):
            return sep_by(' ', attrs)

        cbool = 'chpl_bool'
        int32 = 'int32_t'
        int64 = 'int64_t'
        fcomplex = '_complex64'
        dcomplex = '_complex128'

        val = self.generate_non_tuple(node, scope)
        if val <> None:
            return val

        with match(node):
            if (ir.fn_defn, Attrs, (ir.primitive_type, 'void'), Name, Args, Body, DocComment):
                new_scope('%s%sproc %s(%s) {'%
                          (gen_comment(DocComment),
                           gen_attrs(Attrs),
                           gen(Name), gen_comma_sep(Args)),
                          Body,
                          '}')
                new_def('')

            elif (ir.fn_defn, Attrs, Type, Name, Args, Body, DocComment):
                new_scope('%s%sproc %s(%s): %s {'%
                          (gen_comment(DocComment),
                           gen_attrs(Attrs),
                           gen(Name), gen_comma_sep(Args),
                           gen(Type)),
                          Body,
                          '}')
                new_def('')

            elif (ir.fn_decl, Attrs, (ir.primitive_type, 'void'), Name, Args, DocComment):
                new_def('proc %s(%s)'% (gen(Name), gen_comma_sep(Args)))

            elif (ir.fn_decl, Attrs, Type, Name, Args, DocComment):
                new_def('proc %s(%s): %s'%
                        (gen(Name), gen_comma_sep(Args), gen(Type)))

            elif (ir.call, (ir.deref, (ir.get_struct_item, S, _, (ir.struct_item, _, Name))), Args):
                # We can't do a function pointer call in Chapel
                # Emit a C stub for that
                _, s_id, _, _ = S
                retval_arg = generate_method_stub(scope, node, s_id)
                stubname = '_'.join([epv_qname(s_id),re.sub('^f_', '', Name),'stub'])
                if retval_arg:
                    scope.pre_def(gen(ir.Var_decl(retval_arg, '_retval')))
                    scope.pre_def(gen(ir.Assignment('_retval', ir.Call(stubname, Args+retval_arg))))
                    return '_retval'
                else:
                    return gen(ir.Call(stubname, Args))

            elif (ir.call, (ir.get_struct_item, S, _, (ir.struct_item, _, Name)), Args):
                # We can't do a function pointer call in Chapel
                # Emit a C stub for that
                _, s_id, _, _ = S
                retval_arg = generate_method_stub(scope, node, s_id)
                stubname = '_'.join([epv_qname(s_id), re.sub('^f_', '', Name),'stub'])
                if retval_arg:
                    scope.pre_def(gen(ir.Var_decl(retval_arg, '_retval')))
                    scope.pre_def(gen(ir.Assignment('_retval', ir.Call(stubname, Args+retval_arg))))
                    return '_retval'
                else:
                    return gen(ir.Call(stubname, Args))

            elif (ir.new, Type, Args):
                return 'new %s(%s)'%(gen(Type), gen_comma_sep(Args))

            elif (ir.const, Type):
                return '/*FIXME: CONST*/'+gen(Type)

            # Special handling of rarray types
            elif (ir.arg, Attrs, Mode, (sidl.rarray, Scalar_type, Dimension, Extents), Name):
                (arg_mode, arg_name) = (gen(Mode), gen(Name))
                # rarray type will include a new domain variable definition
                arg_type = '[?_babel_dom_%s] %s'%(Name, gen(Scalar_type))
                return '%s %s: %s'%(arg_mode, arg_name, arg_type)

            elif (ir.arg, Attrs, Mode, Type, Name):
                return '%s %s: %s'%(gen(Mode), gen(Name), gen(Type))

            elif (sidl.class_, (ir.scoped_id, Prefix, Name, Ext), Extends, Implements, Invariants, Methods, DocComment):
                return '%sclass %s' % (gen_comment(DocComment), 
                                       '.'.join(Prefix+['_'.join(Prefix+[Name+Ext])]))

            elif (sidl.array, [], [], []):
                print '** WARNING: deprecated rule use a sidl__array struct instead'
                return 'sidl.Array(opaque, sidl__array)'

            elif (sidl.array, Scalar_type, Dimension, Orientation):
                print '** WARNING: deprecated rule use a sidl__array struct instead'
                if Scalar_type[0] == ir.scoped_id:
                    ctype = 'BaseInterface'
                else:
                    ctype = Scalar_type[1]
                #scope.cstub.optional.add('#include <sidl_%s_IOR.h>'%ctype)
                return 'sidl.Array(%s, sidl_%s__array)'%(gen(Scalar_type), ctype)

            elif (ir.pointer_type, (ir.const, (ir.primitive_type, ir.char))):
                return "string"

            elif (ir.pointer_type, (ir.primitive_type, ir.void)):
                return "opaque"

            elif (ir.pointer_type, Type):
                # ignore wrongfully introduced pointers
                # -> actually I should fix generate_method_stub instead
                return gen(Type)

            elif (ir.typedef_type, cbool):
                return "bool"

            elif (ir.typedef_type, 'sidl_bool'):
                return "bool"

            elif (ir.typedef_type, int32):
                return "int(32)"

            elif (ir.typedef_type, int64):
                return "int(64)"

            elif (ir.typedef_type, fcomplex):
                return "complex(64)"

            elif (ir.typedef_type, dcomplex):
                return "complex(128)"

            elif (ir.struct, (ir.scoped_id, Prefix, Name, Ext), Items, DocComment):
                #print 'prefix %s, name %s, ext %s' %(Prefix, Name, Ext)
                return '.'.join(Prefix+['_'.join(Prefix+[Name+Ext])])

            elif (ir.struct, Name, Items, DocComment):
                # special rule for handling SIDL arrays
                scalar_t = self.is_sidl_array(Name)
                if scalar_t:
                    #scope.cstub.optional.add('#include <sidl_%s_IOR.h>'%ctype)
                    return 'sidl.Array(%s, %s)'%(scalar_t, Name)
                else:
                    # some other struct
                    if Name[0] == '_' and Name[1] == '_': 
                        # by convention, Chapel generates structs as
                        # struct __foo { ... } foo;
                        return Name[2:] 
                    return Name

            elif (ir.get_struct_item, _, (ir.deref, StructName), (ir.struct_item, _, Item)):
                return "%s.%s"%(gen(StructName), gen(Item))

            elif (ir.set_struct_item, _, (ir.deref, StructName), (ir.struct_item, _, Item), Value):
                return gen(StructName)+'.'+gen(Item)+' = '+gen(Value)

            elif (ir.type_decl, (ir.struct, Name, Items, DocComment)):
                itemdecls = gen_semicolon_sep(map(lambda i: ir.Var_decl(i[1], i[2]), Items))
                return gen_comment(DocComment)+str(new_scope1('record %s {\n'%gen(Name), 
                                                              itemdecls, '\n}'))

            elif (ir.var_decl, Type, Name):
                return scope.new_header_def('var %s: %s'%(gen(Name), gen(Type)))

            elif (ir.var_decl_init, (ir.typedef_type, "inferred_type"), Name, Initializer):
                return scope.new_header_def('var %s = %s'%(gen(Name), gen(Initializer)))

            elif (ir.var_decl_init, Type, Name, Initializer):
                return scope.new_header_def('var %s: %s = %s'%
                                            (gen(Name), gen(Type), gen(Initializer)))

            elif (ir.enum, Name, Items, DocComment): return gen(Name)

            elif (ir.type_decl, (ir.enum, Name, Items, DocComment)):
                # Manually transform the Items
                enum_transformed = False
                used_states = []
                for loop_item in Items:
                    if (len(loop_item) == 3):
                        used_states.append(loop_item[2])
                    else:
                        enum_transformed = True

                items_to_use = Items
                if enum_transformed:
                    # Explicitly initialize the enum values, since
                    # Chapel enums start at 1
                    new_items = []
                    avail_state = 0
                    for loop_item in Items:
                        if (len(loop_item) == 3):
                            new_items.append(loop_item)
                        else:
                            while avail_state in used_states:
                                avail_state = avail_state + 1
                            new_items.append(ir.Enumerator_value(loop_item[1], avail_state))
                            used_states.append(avail_state)
                    items_to_use = new_items

                return new_scope1('enum %s {'%gen(Name), gen_comma_sep(items_to_use), '}')

            elif (ir.import_, Name): new_def('use %s;'%Name)


            elif (ir.float, N):   
                return str(N)+':real(32)'

            elif (ir.double, N):   
                return str(N)+':real(64)'

            elif (sidl.custom_attribute, Id):       return gen(Id)
            elif (sidl.method_name, Id, Extension): return gen(Id) + gen(Extension)
            elif (sidl.scoped_id, Prefix, Name, Ext):
                return '.'.join(Prefix+[Name])

            else:
                return super(ChapelCodeGenerator, self).generate(node, scope)
        return scope
