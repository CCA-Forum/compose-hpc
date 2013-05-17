#
# module implementing the basic Annotated Term format as 
# specified at:
#
#  http://www.program-transformation.org/Tools/ATermFormat
#
# matt@galois.com
#

class ATerm:
    def __init__(self):
        self.annotation = None
        self.children = None

    def annotate(self, t):
        self.annotation = t

    def toString(self):
        if (self.annotation != None):
            return "{ " + self.annotation.toString() + " }"
        else:
            return ""

class ATConstant(ATerm):
    def setValue(self, v):
        self.value = v

    def toString(self):
        return str(self.value)+(ATerm.toString(self))

class ATConstructor(ATerm):
    def __init__(self):
        ATerm.__init__(self)
        self.children = []

    def setConstructor(self, cname, kids):
        self.name = cname
        self.children = kids

    def setName(self, name):
        self.name = name

    def addChild(self, kid):
        self.children.append(kid)

    def toString(self):
        s = str(self.name)+"("
        for kid in self.children:
            s += kid.toString()+","
        if len(self.children) > 0:
            s = s[:len(s)-1]
        s += ") "+ATerm.toString(self)
        return s

class ATTuple(ATerm):
    def __init__(self):
        ATerm.__init__(self)
        self.children = []

    def setValue(self, kids):
        self.children = kids

    def addChild(self, kid):
        self.children.append(kid)

    def toString(self):
        s = "("
        for kid in self.children:
            s += kid.toString()+","
        if len(self.children) > 0:
            s = s[:len(s)-1]
        s += ") "+ATerm.toString(self)
        return s

class ATList(ATerm):
    def __init__(self):
        ATerm.__init__(self)
        self.children = []

    def setValue(self, kids):
        self.children = kids

    def addChild(self, kid):
        self.children.append(kid)

    def toString(self):
        s = "["
        for kid in self.children:
            s += kid.toString()+","
        if len(self.children) > 0:
            s = s[:len(s)-1]
        s += "] "+ATerm.toString(self)
        return s

class ATString(ATerm):
    def setValue(self, v):
        self.value = v

    def toString(self):
        s = str(self.value)+ATerm.toString(self)
        return s

class ATInt(ATerm):
    def setValue(self, v):
        self.value = v

    def toString(self):
        s = str(self.value)+ATerm.toString(self)
        return s

class ATReal(ATerm):
    def setValue(self, v):
        self.value = v

    def toString(self):
        s = str(self.value)+ATerm.toString(self)
        return s
