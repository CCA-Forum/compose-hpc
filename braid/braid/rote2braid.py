#!/usr/bin/env python
# -*- python -*-
## @package braid
#
# Command line handling for the BRAID tool
#
# Please report bugs to <adrian@llnl.gov>.
#
# \authors <pre>
#
# Copyright (c) 2012, Lawrence Livermore National Security, LLC.
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

from codegen import *

@matcher(globals(), debug=False)
def do(node):
    with match(node):
        if   ('project', Files, _, _):      return do(Files)
        elif ('source_file', Global, _, _): return do(Global)
        elif ('global', Defs, _, _):        return do(Defs)

        elif ('class_declaration', Defn, ('class_declaration_annotation', Name, _, _, _), _):
            return ir.Struct(Name, do(Defn), '')

        elif ('class_definition', Decls, _, _): 
            return []
            return [Struct_item(Name, do(Decl), '') for Decl in Decls]

        elif ('function_declaration', Params, Def, _, ('function_declaration_annotation', Type, Name, _, _, _), _): 
            return [ir.Fn_decl([], do(Type), Name, do(Params), ''), do(Def)]

        elif ('function_definition', Bb, _, _): return ir.Fn_defn(do(Bb))
        elif ('function_parameter_list', Params, _, _): return do(Params)
        elif ('initialized_name', _, ('initialized_name_annotation', Type, Name, _, _, _), _): 
            # this needs more work
            return ir.Arg([], ir.inout, do(Type), Name)

        elif ('function_type', Type, _, _): return do(Type)
        elif ('typedef_declaration', _, ('typedef_annotation', Name, Type, _), _): 
            return ir.Typedef_type(Name)
        elif ('typedef_type', Name, Type): return ir.Typedef_type(Name)

        elif ('basic_block', Stmts, _, _): return do(Stmts)
        elif ('expr_stmt', Expr, _, _): return ir.stmt(do(Expr))
        elif ('function_call_exp', Fn, Args, _, _): return ir.call(do(Fn), do(Args))
        elif ('function_ref_exp', ('function_ref_exp_annotation', Name, _, _, _), _): return Name

        elif ('pointer_type', Type): return ir.Pointer_type(do(Type))
        elif ('modifier_type', Type, ('type_modifier', _, _, 'const', _)): return ir.Const(do(Type))

        elif ('modifier_type', Type, _): return do(Type) # precision loss
        elif ('class_type', Name, _, _): return ir.Struct(ir.Scoped_id([], Name, ''), [], '')
        elif 'type_bool': return ir.pt_bool
        elif 'type_char': return ir.pt_char
        elif 'type_int':  return ir.pt_int
        elif 'type_long': return ir.pt_long
        elif 'type_void': return ir.pt_void
        elif 'type_float':  return ir.pt_float
        elif 'type_double': return ir.pt_double
        elif 'type_long_double': return ir.pt_long_double
        elif 'type_long_long': return ir.pt_long_long
        elif ('array_type', Type, ('unsigned_long_val', ('value_annotation', Dim, _), _), _, _):
            return ir.Pointer_type(do(Type))
        elif ('array_type', Type, _, _, _):
            return ir.Pointer_type(do(Type))
        elif 'null': return []
        elif A:
            if isinstance(A, list):
                return map(do, A)
            else:
                if isinstance(A, str):
                    if re.match(r'^type_.*$', A):
                        return ir.Typedef_type(A)
                print 'unhandled node', A, 'len=', len(A)
                #import pdb; pdb.set_trace()


if __name__ == '__main__':
    import argparse, config
    cmdline = argparse.ArgumentParser(description=config.PACKAGE_STRING+'''
Convert the ROTE intermediate representation (as generated by
minitermite's src2term) into BRAID IR.
''',
    epilog="Please report bugs to <%s>."%config.PACKAGE_BUGREPORT)

    cmdline.add_argument('term_files', metavar='<file.term>', nargs='*',#type=file
			 help='Term file to use as input')

    cmdline.add_argument('--version', action='store_true', help='print version and exit')
    cmdline.add_argument('-v', '--verbose', action='store_true', help='print more debug info')    

    args = cmdline.parse_args()

    if args.version:
        print config.PACKAGE_STRING
        print '# This version of BRAID was configured on %s, %s.'%(
            config.BUILD_MACHINE,config.CONFIGURE_DATE)
        exit(0)
    if len(args.term_files) == 0:
        cmdline.print_help()
        exit(1)

    import parse_tree as pt
    for term_file in args.term_files:
        rote_ir = pt.parse_tree_file(term_file)
        print repr(do(rote_ir))
