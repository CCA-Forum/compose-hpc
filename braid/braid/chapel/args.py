#!/usr/bin/env python
# -*- python -*-
## @package chapel.args
#
# High-level argument conversion IOR <-> Chapel
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

import ir, sidl
from patmat import *

char_lut = '''
/* This burial ground of bytes is used for char [in]out arguments. */
static const unsigned char chpl_char_lut[512] = {
  '''+' '.join(['%d, 0,'%i for i in range(0, 256)])+'''
};
'''

def ir_arg_to_chpl((arg, attrs, mode, typ, name)):
    return arg, attrs, mode, ir_type_to_chpl(typ), name

def ir_type_to_chpl(typ):
    mapping = {
        ir.pt_void:     ir.pt_void,
        ir.pt_bool:     ir.Typedef_type('_Bool'),
        ir.pt_string:   ir.const_str,
        ir.pt_int:      ir.Typedef_type('int32_t'),
        ir.pt_long:     ir.Typedef_type('int64_t'),
        ir.pt_char:     ir.const_str,
        ir.pt_fcomplex: ir.Typedef_type('_complex64'), 
        ir.pt_dcomplex: ir.Typedef_type('_complex128'),
        ir.pt_float:    ir.pt_float,
        ir.pt_double:   ir.pt_double,
        sidl.pt_opaque: ir.Pointer_type(ir.pt_void)
        }
    try:
        return mapping[typ]
    except:
        if typ[0] == ir.enum: 
            return ir.Typedef_type('int64_t')
        return typ
   

def chpl_to_ior(convs, optional, (arg, attrs, mode, typ, name)):
    """
    Extract name and generate argument conversions by appending convs
    """
    from cgen import c_gen
    deref = ref = ''
    accessor = '.'
    if mode <> sidl.in_ and name <> '_retval':
        deref = '*'
        ref = '&'
        accessor = '->'

    # Case-by-case for each data type

    # BOOL
    if typ == sidl.pt_bool:
        convs.append(ir.Comment('sidl_bool is an int, but chapel bool is a char/_Bool'))
        convs.append((ir.stmt, '_proxy_{n} = ({typ}){p}{n}'
                      .format(n=name, p=deref, typ=c_gen(sidl.pt_bool))))

    # CHAR
    elif typ == sidl.pt_char:
        convs.append(ir.Comment('in chapel, a char is a string of length 1'))

        convs.append((ir.stmt, '_proxy_{n} = (int){p}{n}[0]'
                         .format(n=name, p=deref)))

        optional.add(char_lut)

    # COMPLEX - 32/64 Bit components
    elif (typ == sidl.pt_fcomplex or typ == sidl.pt_dcomplex):
        fmt = {'n':name, 'a':accessor}
        convs.append((ir.stmt, '_proxy_{n}.real = {n}{a}re'.format(**fmt)))
        convs.append((ir.stmt, '_proxy_{n}.imaginary = {n}{a}im'.format(**fmt)))

    elif typ[0] == sidl.enum:
        # No special treatment for enums, rely on chpl runtime to set it
        convs.append(ir.Stmt(ir.Assignment('_proxy_'+name, ir.Sign_extend(64, name))))
        pass

    # ARRAYS
    elif typ[0] == sidl.array: # Scalar_type, Dimension, Orientation
        import pdb; pdb.set_trace()

    # We should find a cleaner way of implementing this
    if name == 'self' and member_chk(ir.pure, attrs):
        convs.append(ir.Stmt(ir.Assignment('_proxy_'+name, '(({0})((struct sidl_BaseInterface__object*)self)->d_object)'.format(c_gen(typ)))))


def ior_to_chpl(convs, optional, (arg, attrs, mode, typ, name)):
    """
    Extract name and generate argument conversions by appending convs
    """
    from cgen import c_gen
    deref = ref = ''
    accessor = '.'
    if mode <> sidl.in_ and name <> '_retval':
        deref = '*'
        ref = '&'
        accessor = '->'

    # Case-by-case for each data type

    # BOOL
    if typ == sidl.pt_bool:
        convs.append(ir.Comment(
            'sidl_bool is an int, but chapel bool is a char/_Bool'))
        convs.append((ir.stmt, '{p}{n} = ({typ})_proxy_{n}'
                          .format(p=deref, n=name, typ=c_gen(ir_type_to_chpl(sidl.pt_bool)))))

        # Bypass the bool -> int conversion for the stub decl
        #typ = (ir.typedef_type, '_Bool')

    # CHAR
    elif typ == sidl.pt_char:
        convs.append(ir.Comment(
            'in chapel, a char is a string of length 1'))
        typ = ir.const_str

        # we can't allocate a new string, this would leak memory
        convs.append((ir.stmt,
                          '{p}{n} = (const char*)&chpl_char_lut[2*(unsigned char)_proxy_{n}]'
                          .format(p=deref, n=name)))
        optional.add(char_lut)

    # STRING
    elif typ == sidl.pt_string:
        typ = ir.const_str
        # Convert null pointer into empty string
        convs.append((ir.stmt, 'if ({p}{n} == NULL) {p}{n} = ""'
                          .format(n=name, p=deref))) 

    # INT
    elif typ == sidl.pt_int:
        typ = ir.Typedef_type('int32_t')

    # LONG
    elif typ == sidl.pt_long:
        typ = ir.Typedef_type('int64_t')

    # COMPLEX - 32/64 Bit components
    elif (typ == sidl.pt_fcomplex or typ == sidl.pt_dcomplex):
        fmt = {'n':name, 'a':accessor}
        convs.append((ir.stmt, '{n}{a}re = _proxy_{n}.real'.format(**fmt)))
        convs.append((ir.stmt, '{n}{a}im = _proxy_{n}.imaginary'.format(**fmt)))

    elif typ[0] == sidl.enum:
        # No special treatment for enums, rely on chpl runtime to set it
        convs.append(ir.Stmt(ir.Assignment('_proxy_'+name, ir.Sign_extend(64, name))))
        pass

    # ARRAYS
    elif typ[0] == sidl.array: # Scalar_type, Dimension, Orientation
        import pdb; pdb.set_trace()
