###
### python AST to aterms.  uses the python asdl specification from
### the language source distribution to define the term structure
### to mimic the AST structure.
###
### experimental
###
### matt@galois.com
###

#
# make sure the asdl.py module from the python source distribution is in
# your python path.  E.g.:
#
#   export PYTHONPATH=/Users/matt/tmp/Python-2.7.4/Parser
#
import asdl
import sys

filename = "/Users/matt/tmp/Python-2.7.4/Parser/Python.asdl"
#filename = "/Users/matt/tmp/Python-3.3.1/Parser/Python.asdl"


#
# boilerplate header that includes the AST module and Aterm Module,
# as well setting up the handler dictionary
#
boilerplate = """import ast
import aterm
import sys

handler = {}

def straterm(x):
    t = aterm.ATConstant()
    t.setValue(str(x))
    return t

handler['str'] = straterm
handler['int'] = straterm
handler['NoneType'] = straterm

def handle(x):
    if x.__class__.__name__ in handler:
        return handler[x.__class__.__name__](x)
    else:
        # print "Error: No supported handler for "+x.__class__.__name__
        return straterm(x)
"""

#
# read python ASDL file
#
mod = asdl.parse(filename)

# emit boilerplate
print boilerplate

def handleConstructor(c):
    handleProduct(c.name,c)

def handleProduct(n,c):
    name = str(n)
    fields = c.fields

    print "def handle_"+str(name)+"(x):"
    print "    t = aterm.ATConstructor()"
    print "    t.setName(\""+name+"\")"
    for field in fields:
        fname = str(field.name)
        seq = field.seq
        opt = field.opt
        if opt:
            print "    #OPTIONAL"
            print "    if hasattr(x,'"+fname+"'):"
            print "        "+fname+" = x."+fname
            print "        t.addChild(handle("+fname+"))"
            print "    else:"
            print "        tkid = aterm.ATConstant()"
            print "        tkid.setValue('None')"
            print "        t.addChild(tkid)"
        elif seq:
            print "    if hasattr(x,'"+fname+"'):"
            print "        "+fname+" = x."+fname
            print "        tkid = aterm.ATList()"
            print "        for elt in "+fname+":"
            print "            tkid.addChild(handle(elt))"
            print "        t.addChild(tkid)"
        else:
            print "    "+fname+" = x."+fname
            print "    t.addChild(handle("+fname+"))"
        print

    print "    return t"
    print 
    print "handler['"+name+"'] = handle_"+name
    print
    print "# end "+name
    print


#
# code generate: for each definition, process it.
#
for defn in mod.dfns:
    name = defn.name
    v = defn.value

    if type(v) == asdl.Product:
        handleProduct(name,v)
    elif type(v) == asdl.Sum:
        ts = v.types
        for t in ts:
            handleConstructor(t)
    else:
        print "ERROR: Unsupported definition type"
        sys.exit(0)

#
# print closing boiler plate
#
print """

if (len(sys.argv) == 1):
    print "Specify a target file."
    sys.exit(1)


f = open(sys.argv[1],'r')
s = f.read()
f.close()


tree = ast.parse(s)

t = handle(tree)

print t.toString()
"""
