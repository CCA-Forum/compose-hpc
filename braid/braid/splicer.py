#!/usr/bin/env python
# -*- python -*-
## @package splicer
# Support functions for handling Babel Splicer blocks
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
import os,re

def replace(filename, splicer_name, text):
    """
    Replace the contents of one specific \c splicer_block with \c text.
    \param filename      The name of the file to edit.
    \param splicer_name  The name of the splicer block to edit.
    \param text          The text to put inside the splicer block.

    expect(string, string, string)
    """
    #print "splicing", splicer_name, "with '"+text+"'"
    if text == None: raise Exception("Empty splicer text")

    # first make a backup of the old file
    os.rename(filename, filename+'~')
    dest = open(filename, 'w')
    src = open(filename+'~', 'r')

    inside = False
    did_replace = False
    for line in src:
        if re.match(r'.*DO-NOT-DELETE splicer\.begin\('+splicer_name+r'\).*', 
                    line):
            dest.write(line)
            dest.write(text+'\n')
            inside = True
            did_replace = True
        elif (inside and
            re.match(r'.*DO-NOT-DELETE splicer\.end\('+splicer_name+r'\).*', 
                     line)):
                inside = False

        if not inside:
            dest.write(line)
                
    if inside:
        raise Exception("unclosed splicer block")

    if not did_replace:
        raise Exception("splicer block not found")

    src.close()
    dest.close()


def record(filename):
    """
    Return a dict with the contents of all splicer blocks in the file
    \c filename.
    """
    src = open(filename, 'r')
    inside = False
    splicer_name = ''
    splicers = {}
    for line in src:
        m = re.match(r'.*DO-NOT-DELETE splicer\.begin\((.*)\).*', line)
        if m:
            splicer_name = m.group(1)
            # print "splicer_block(%s)", splicer_name
            splicer = []
            inside = True
        elif (inside and re.match(
                r'.*DO-NOT-DELETE splicer\.end\(%s\).*'%splicer_name, line)):
            inside = False
            splicers[splicer_name] = splicer
        elif inside:
            # store it and remove the trailing newline
            splicer.append(re.sub(r'\r?\n$', '', line[:-1]))

    if inside:
        raise Exception("unclosed splicer block: "+splicer_name)
    
    src.close()
    return splicers

def apply_all(filename, lines, splicers):
    """
    Apply the previously recorded splicers \c splicers to a list of lines
    """
    dest = []
    all_splicers = set(splicers)
    splicer_name = ''
    inside = False
    for line in lines:
        m = re.match(r'.*DO-NOT-DELETE splicer\.begin\((.*)\).*', line)
        if m:
            dest.append(line)
            splicer_name = m.group(1)
            try:
                block = splicers[splicer_name]                
                for l in block:
                    dest.append(l)
            except KeyError:
                if len(splicers) > 0: # be quiet if we created a new file                    
                    print "**INFO: The following new splicer block was added to %s: %s" \
                        % (filename, splicer_name)
            inside = True

        elif (inside and re.match(
                r'.*DO-NOT-DELETE splicer\.end\(%s\).*'%splicer_name, line)):
            inside = False
            try: 
                all_splicers.remove(splicer_name)
            except KeyError:
                # this is the closing of a new splicer block
                pass

        if not inside:
            dest.append(line)                

    # error reporting
    if inside:
        raise Exception("unclosed splicer block: "+splicer_name)

    if len(all_splicers) > 0:
        print "**WARNING: The following splicer blocks are no longer present in %s: " % filename
        for name in all_splicers: 
            print name
            if splicers[name]:
                print splicers[name]
                dest.append('ORPHANED SPLICER BLOCK splicer.begin(%s)'%name)
                map(dest.append, splicers[name])
                dest.append('ORPHANED SPLICER BLOCK splicer.end(%s)'%name)
                
    return '\n'.join(dest)
