#!/usr/bin/env python
# -*- python -*-
## @package sidlobjects
#
# Object-oriented representation of SIDL entities.
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
# This file is part of BRAID ( http://compose-hpc.sourceforge.net/ ).
# Please read the COPYRIGHT file for Our Notice and for the BSD License.
#
# </pre>
#

import sidl
from sidl_symbols import visit_hierarchy, scan_methods
from patmat import member_chk
from utils import accepts, returns

@accepts(object, tuple)
def make_extendable(symbol_table, sidl_ext):
    if sidl.is_scoped_id(sidl_ext): 
        return make_extendable(symbol_table, symbol_table[sidl_ext])

    if sidl.is_class(sidl_ext): 
        return Class(symbol_table, sidl_ext, [])

    if sidl.is_interface(sidl_ext): 
        return Interface(symbol_table, sidl_ext, [])

    import pdb; pdb.set_trace()



class Extendable(object):
    """
    Base class for Class, Interface, ...
    """
    def __init__(self, symbol_table, ext, attrs):
        """
        Create a new Extendable object from the sidl tuple representation.
        """
        self.data = ext
        self.symbol_table = symbol_table
        self.is_abstract = member_chk(sidl.abstract, attrs)
        if sidl.is_scoped_id(ext[1]):
            self.name = ext[1][2]
            self.qualified_name = ext[1][1]+[self.name+ext[1][3]]
        else:
            self.name = ext[1]
            self.qualified_name = symbol_table.prefix+[self.name]
        self.qualified_name = symbol_table.prefix+[self.name]
        self.extends = ext[2]
        self.implements = [] # default for interface

        # invoke scan_methods() to populate these
        self.all_methods = None 
        self.has_static_methods = None

    def get_parent(self):
        """
        return the base class/interface of \c class_or_interface
        """
        extends = self.data[2]
        if extends == []:
            return None
        ext = self.symbol_table[extends[0][1]]

        return make_extendable(self.symbol_table, ext)

    def get_parents(self, all_parents):
        """
        Return a list of the names of all base classes and implemented
        interfaces of a class in \c all_parents
        """

        def f(s_id):
            c_i = self.symbol_table[s_id]
            all_parents.append(c_i)

        start = sidl.Scoped_id(self.symbol_table.prefix, sidl.type_id(self.data), '')
        visit_hierarchy(start, f, self.symbol_table, [])
        return all_parents[1:]

    def get_parent_interfaces(self):

        """
        return a list of the scoped ids of all implemented interfaces
        """
        isinterface = sidl.is_interface(self.data)
        isclass = sidl.is_class(self.data)
        assert isclass or isinterface

        def f(s_id):
            c_i = self.symbol_table[s_id]
            if c_i[0] == sidl.interface:
                # make the scoped id hashable by converting the list
                # of modules into a tuple
                sid, modules, name, ext = s_id
                all_interfaces.append((sid, tuple(modules), name, ext))

        all_interfaces = []
        #tid = sidl.type_id(self.data)
        #if not isinstance(tid, tuple):
        #    start = sidl.Scoped_id(self.symbol_table.prefix, tid, '')
        #else:
        #    start = tid
        start = self.get_scoped_id()

        visit_hierarchy(start, f, self.symbol_table, [])
        if isinterface:
            # first one is the interface itself
            return all_interfaces[1:]
        return all_interfaces

    def has_parent_interface(self, intf):
        """
        return \c True if this extendable implements the interface \c intf.
        """
        return intf in self.get_parent_interfaces()

    def get_direct_parent_interfaces(self):

        """
        return a list of all direct (local) implemented interfaces
        """

        if self.data[0] == sidl.interface:
            parents = sidl.interface_extends(self.data)
        else:
            parents = sidl.class_implements(self.data)
        return [self.symbol_table[impl] for _, impl in parents]

    def get_unique_interfaces(self):
        """
        Extract the unique interfaces from this extendable object.
        The unique interfaces are those that belong to this class but
        do not belong to one of its parents (if they exit).  The
        returned set consists of objects of the type
        <code>Interface</code>.
        """
        unique = set(self.get_parent_interfaces())
        parent = self.get_parent()
        if parent:
            unique -= set(parent.get_parent_interfaces())
        return unique

    def scan_methods(self):
        """
        Recursively resolve the inheritance hierarchy and build a list of
        all methods in their EPV order.

        This function populates the \c all_methods and the \c
        has_static_methods fields.
        """
        all_names = set()
        self.all_methods = []
        # Consolidate all methods, defined and inherited
        scan_methods(self.symbol_table, 
                     self.is_abstract, 
                     self.extends, 
                     self.implements, 
                     self.get_methods(), 
                     all_names, 
                     self.all_methods, 
                     self, 
                     True)

        self.has_static_methods = any(map(
                lambda m: member_chk(sidl.static, sidl.method_method_attrs(m)), 
                self.all_methods))

    def number_of_methods(self):
        return len(sidl.ext_methods(self.data))

    def is_interface(self):
        return sidl.is_interface(self.data)

    def is_class(self):
        return sidl.is_class(self.data)

    def get_id(self):
        """
        Same as \ref sidl.type_id()
        """
        return self.name #sidl.type_id(self.data)


    @returns(tuple)
    def get_scoped_id(self):
        """
        Same as \ref sidl.get_scoped_id()
        """
        return sidl.get_scoped_id(self.symbol_table, self.data)

class Class(Extendable):
    """
    A wrapper around the sidl class entity.
    """

    def __init__(self, symbol_table, sidl_class, attrs):
        """
        Create a new Class object from the sidl tuple representation.
        """
        super(Class, self).__init__(symbol_table, sidl_class, attrs)
        self.implements = sidl.class_implements(sidl_class)
        self.doc_comment = sidl.class_doc_comment(sidl_class)

    def is_class(self):
        return True

    def is_interface(self):
        return False

    def get_methods(self):
        return sidl.class_methods(self.data)


class Interface(Extendable):
    """
    A wrapper around the sidl interface entity.
    """

    def __init__(self, symbol_table, sidl_interface, attrs):
        """
        Create a new Class object from the sidl tuple representation.
        """
        super(Interface, self).__init__(symbol_table, sidl_interface, attrs)
        self.doc_comment = sidl.interface_doc_comment(sidl_interface)

    def is_interface(self):
        return True

    def is_class(self):
        return False

    def get_methods(self):
        return sidl.interface_methods(self.data)
