#!/usr/bin/env python
# -*- python -*-
## @package upc.cgen
#
# BRAID code generator implementation for UPC
#
# Please report bugs to <components@llnl.gov>.
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
#
import ir, sidlir, re
import ior
from patmat import *
from utils import *
from codegen import (
    ClikeCodeGenerator, CCodeGenerator, c_gen,
    SourceFile, CFile, CCompoundStmt, Scope, generator, accepts,
    sep_by
)
#import upc_conversions as conv

class UPCFile(CFile):
    """
    A BRAID-style code generator output file manager for the UPC language.

    * UPC files also have a cstub which is used to output code that
      can not otherwise be expressed in UPC.

    * The main_area member denotes the space that defaults to the
      module's main() function.
    """

    @accepts(object, str, object, int)
    def __init__(self, name="", parent=None, relative_indent=0):
        super(UPCFile, self).__init__(name, parent, relative_indent)

    def write(self):
        """
        Atomically write the UPCFile and its header to disk, using the
        basename provided in the constructor.
        Empty files will not be created.
        """
        cname = self._name+'.upc'
        hname = self._name+'.h'
        if self._defs:   write_to(cname, self.dot_c())
        if self._header: write_to(hname, self.dot_h(hname))


class UPCCodeGenerator(CCodeGenerator):
    """
    A BRAID-style code generator for UPC.
    """
    pass

def upc_gen(ir, scope=None):
    """
    Generate UPC code with the optional scope argument
    \return a string
    """
    if scope == None:
        scope = CScope()
    return str(UPCCodeGenerator().generate(ir, scope))
