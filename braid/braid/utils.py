#!/usr/bin/env python
# -*- python -*-
## @package lists
#
# Some often use list-manipulation functions and type-checking
# decorators.
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
import itertools

def unzip(lst):
    """
    Like zip(*lst), but works also for empty lists.
    """
    return [a for a, b in lst], [b for a, b in lst]

def unzip3(lst):
    """
    unzip for three elements
    """
    return [a for a, b, c in lst], [b for a, b, c in lst], [c for a, b, c in lst]

def flatten2d(lst):
    return list(itertools.chain.from_iterable(lst))

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
    Similar to \c string.join() but appends the separator also after
    the last element, if any.
    """
    if len(strings) > 0:
        return separator.join(strings+[''])
    else:
        return ''
