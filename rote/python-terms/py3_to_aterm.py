import ast
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

def handle_Module(x):
    t = aterm.ATConstructor()
    t.setName("Module")
    if hasattr(x,'body'):
        body = x.body
        tkid = aterm.ATList()
        for elt in body:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['Module'] = handle_Module

# end Module

def handle_Interactive(x):
    t = aterm.ATConstructor()
    t.setName("Interactive")
    if hasattr(x,'body'):
        body = x.body
        tkid = aterm.ATList()
        for elt in body:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['Interactive'] = handle_Interactive

# end Interactive

def handle_Expression(x):
    t = aterm.ATConstructor()
    t.setName("Expression")
    body = x.body
    t.addChild(handle(body))

    return t

handler['Expression'] = handle_Expression

# end Expression

def handle_Suite(x):
    t = aterm.ATConstructor()
    t.setName("Suite")
    if hasattr(x,'body'):
        body = x.body
        tkid = aterm.ATList()
        for elt in body:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['Suite'] = handle_Suite

# end Suite

def handle_FunctionDef(x):
    t = aterm.ATConstructor()
    t.setName("FunctionDef")
    name = x.name
    t.addChild(handle(name))

    args = x.args
    t.addChild(handle(args))

    if hasattr(x,'body'):
        body = x.body
        tkid = aterm.ATList()
        for elt in body:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    if hasattr(x,'decorator_list'):
        decorator_list = x.decorator_list
        tkid = aterm.ATList()
        for elt in decorator_list:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    #OPTIONAL
    if hasattr(x,'returns'):
        returns = x.returns
        t.addChild(handle(returns))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    return t

handler['FunctionDef'] = handle_FunctionDef

# end FunctionDef

def handle_ClassDef(x):
    t = aterm.ATConstructor()
    t.setName("ClassDef")
    name = x.name
    t.addChild(handle(name))

    if hasattr(x,'bases'):
        bases = x.bases
        tkid = aterm.ATList()
        for elt in bases:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    if hasattr(x,'keywords'):
        keywords = x.keywords
        tkid = aterm.ATList()
        for elt in keywords:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    #OPTIONAL
    if hasattr(x,'starargs'):
        starargs = x.starargs
        t.addChild(handle(starargs))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    #OPTIONAL
    if hasattr(x,'kwargs'):
        kwargs = x.kwargs
        t.addChild(handle(kwargs))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    if hasattr(x,'body'):
        body = x.body
        tkid = aterm.ATList()
        for elt in body:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    if hasattr(x,'decorator_list'):
        decorator_list = x.decorator_list
        tkid = aterm.ATList()
        for elt in decorator_list:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['ClassDef'] = handle_ClassDef

# end ClassDef

def handle_Return(x):
    t = aterm.ATConstructor()
    t.setName("Return")
    #OPTIONAL
    if hasattr(x,'value'):
        value = x.value
        t.addChild(handle(value))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    return t

handler['Return'] = handle_Return

# end Return

def handle_Delete(x):
    t = aterm.ATConstructor()
    t.setName("Delete")
    if hasattr(x,'targets'):
        targets = x.targets
        tkid = aterm.ATList()
        for elt in targets:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['Delete'] = handle_Delete

# end Delete

def handle_Assign(x):
    t = aterm.ATConstructor()
    t.setName("Assign")
    if hasattr(x,'targets'):
        targets = x.targets
        tkid = aterm.ATList()
        for elt in targets:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    value = x.value
    t.addChild(handle(value))

    return t

handler['Assign'] = handle_Assign

# end Assign

def handle_AugAssign(x):
    t = aterm.ATConstructor()
    t.setName("AugAssign")
    target = x.target
    t.addChild(handle(target))

    op = x.op
    t.addChild(handle(op))

    value = x.value
    t.addChild(handle(value))

    return t

handler['AugAssign'] = handle_AugAssign

# end AugAssign

def handle_For(x):
    t = aterm.ATConstructor()
    t.setName("For")
    target = x.target
    t.addChild(handle(target))

    iter = x.iter
    t.addChild(handle(iter))

    if hasattr(x,'body'):
        body = x.body
        tkid = aterm.ATList()
        for elt in body:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    if hasattr(x,'orelse'):
        orelse = x.orelse
        tkid = aterm.ATList()
        for elt in orelse:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['For'] = handle_For

# end For

def handle_While(x):
    t = aterm.ATConstructor()
    t.setName("While")
    test = x.test
    t.addChild(handle(test))

    if hasattr(x,'body'):
        body = x.body
        tkid = aterm.ATList()
        for elt in body:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    if hasattr(x,'orelse'):
        orelse = x.orelse
        tkid = aterm.ATList()
        for elt in orelse:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['While'] = handle_While

# end While

def handle_If(x):
    t = aterm.ATConstructor()
    t.setName("If")
    test = x.test
    t.addChild(handle(test))

    if hasattr(x,'body'):
        body = x.body
        tkid = aterm.ATList()
        for elt in body:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    if hasattr(x,'orelse'):
        orelse = x.orelse
        tkid = aterm.ATList()
        for elt in orelse:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['If'] = handle_If

# end If

def handle_With(x):
    t = aterm.ATConstructor()
    t.setName("With")
    if hasattr(x,'items'):
        items = x.items
        tkid = aterm.ATList()
        for elt in items:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    if hasattr(x,'body'):
        body = x.body
        tkid = aterm.ATList()
        for elt in body:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['With'] = handle_With

# end With

def handle_Raise(x):
    t = aterm.ATConstructor()
    t.setName("Raise")
    #OPTIONAL
    if hasattr(x,'exc'):
        exc = x.exc
        t.addChild(handle(exc))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    #OPTIONAL
    if hasattr(x,'cause'):
        cause = x.cause
        t.addChild(handle(cause))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    return t

handler['Raise'] = handle_Raise

# end Raise

def handle_Try(x):
    t = aterm.ATConstructor()
    t.setName("Try")
    if hasattr(x,'body'):
        body = x.body
        tkid = aterm.ATList()
        for elt in body:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    if hasattr(x,'handlers'):
        handlers = x.handlers
        tkid = aterm.ATList()
        for elt in handlers:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    if hasattr(x,'orelse'):
        orelse = x.orelse
        tkid = aterm.ATList()
        for elt in orelse:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    if hasattr(x,'finalbody'):
        finalbody = x.finalbody
        tkid = aterm.ATList()
        for elt in finalbody:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['Try'] = handle_Try

# end Try

def handle_Assert(x):
    t = aterm.ATConstructor()
    t.setName("Assert")
    test = x.test
    t.addChild(handle(test))

    #OPTIONAL
    if hasattr(x,'msg'):
        msg = x.msg
        t.addChild(handle(msg))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    return t

handler['Assert'] = handle_Assert

# end Assert

def handle_Import(x):
    t = aterm.ATConstructor()
    t.setName("Import")
    if hasattr(x,'names'):
        names = x.names
        tkid = aterm.ATList()
        for elt in names:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['Import'] = handle_Import

# end Import

def handle_ImportFrom(x):
    t = aterm.ATConstructor()
    t.setName("ImportFrom")
    #OPTIONAL
    if hasattr(x,'module'):
        module = x.module
        t.addChild(handle(module))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    if hasattr(x,'names'):
        names = x.names
        tkid = aterm.ATList()
        for elt in names:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    #OPTIONAL
    if hasattr(x,'level'):
        level = x.level
        t.addChild(handle(level))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    return t

handler['ImportFrom'] = handle_ImportFrom

# end ImportFrom

def handle_Global(x):
    t = aterm.ATConstructor()
    t.setName("Global")
    if hasattr(x,'names'):
        names = x.names
        tkid = aterm.ATList()
        for elt in names:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['Global'] = handle_Global

# end Global

def handle_Nonlocal(x):
    t = aterm.ATConstructor()
    t.setName("Nonlocal")
    if hasattr(x,'names'):
        names = x.names
        tkid = aterm.ATList()
        for elt in names:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['Nonlocal'] = handle_Nonlocal

# end Nonlocal

def handle_Expr(x):
    t = aterm.ATConstructor()
    t.setName("Expr")
    value = x.value
    t.addChild(handle(value))

    return t

handler['Expr'] = handle_Expr

# end Expr

def handle_Pass(x):
    t = aterm.ATConstructor()
    t.setName("Pass")
    return t

handler['Pass'] = handle_Pass

# end Pass

def handle_Break(x):
    t = aterm.ATConstructor()
    t.setName("Break")
    return t

handler['Break'] = handle_Break

# end Break

def handle_Continue(x):
    t = aterm.ATConstructor()
    t.setName("Continue")
    return t

handler['Continue'] = handle_Continue

# end Continue

def handle_BoolOp(x):
    t = aterm.ATConstructor()
    t.setName("BoolOp")
    op = x.op
    t.addChild(handle(op))

    if hasattr(x,'values'):
        values = x.values
        tkid = aterm.ATList()
        for elt in values:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['BoolOp'] = handle_BoolOp

# end BoolOp

def handle_BinOp(x):
    t = aterm.ATConstructor()
    t.setName("BinOp")
    left = x.left
    t.addChild(handle(left))

    op = x.op
    t.addChild(handle(op))

    right = x.right
    t.addChild(handle(right))

    return t

handler['BinOp'] = handle_BinOp

# end BinOp

def handle_UnaryOp(x):
    t = aterm.ATConstructor()
    t.setName("UnaryOp")
    op = x.op
    t.addChild(handle(op))

    operand = x.operand
    t.addChild(handle(operand))

    return t

handler['UnaryOp'] = handle_UnaryOp

# end UnaryOp

def handle_Lambda(x):
    t = aterm.ATConstructor()
    t.setName("Lambda")
    args = x.args
    t.addChild(handle(args))

    body = x.body
    t.addChild(handle(body))

    return t

handler['Lambda'] = handle_Lambda

# end Lambda

def handle_IfExp(x):
    t = aterm.ATConstructor()
    t.setName("IfExp")
    test = x.test
    t.addChild(handle(test))

    body = x.body
    t.addChild(handle(body))

    orelse = x.orelse
    t.addChild(handle(orelse))

    return t

handler['IfExp'] = handle_IfExp

# end IfExp

def handle_Dict(x):
    t = aterm.ATConstructor()
    t.setName("Dict")
    if hasattr(x,'keys'):
        keys = x.keys
        tkid = aterm.ATList()
        for elt in keys:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    if hasattr(x,'values'):
        values = x.values
        tkid = aterm.ATList()
        for elt in values:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['Dict'] = handle_Dict

# end Dict

def handle_Set(x):
    t = aterm.ATConstructor()
    t.setName("Set")
    if hasattr(x,'elts'):
        elts = x.elts
        tkid = aterm.ATList()
        for elt in elts:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['Set'] = handle_Set

# end Set

def handle_ListComp(x):
    t = aterm.ATConstructor()
    t.setName("ListComp")
    elt = x.elt
    t.addChild(handle(elt))

    if hasattr(x,'generators'):
        generators = x.generators
        tkid = aterm.ATList()
        for elt in generators:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['ListComp'] = handle_ListComp

# end ListComp

def handle_SetComp(x):
    t = aterm.ATConstructor()
    t.setName("SetComp")
    elt = x.elt
    t.addChild(handle(elt))

    if hasattr(x,'generators'):
        generators = x.generators
        tkid = aterm.ATList()
        for elt in generators:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['SetComp'] = handle_SetComp

# end SetComp

def handle_DictComp(x):
    t = aterm.ATConstructor()
    t.setName("DictComp")
    key = x.key
    t.addChild(handle(key))

    value = x.value
    t.addChild(handle(value))

    if hasattr(x,'generators'):
        generators = x.generators
        tkid = aterm.ATList()
        for elt in generators:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['DictComp'] = handle_DictComp

# end DictComp

def handle_GeneratorExp(x):
    t = aterm.ATConstructor()
    t.setName("GeneratorExp")
    elt = x.elt
    t.addChild(handle(elt))

    if hasattr(x,'generators'):
        generators = x.generators
        tkid = aterm.ATList()
        for elt in generators:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['GeneratorExp'] = handle_GeneratorExp

# end GeneratorExp

def handle_Yield(x):
    t = aterm.ATConstructor()
    t.setName("Yield")
    #OPTIONAL
    if hasattr(x,'value'):
        value = x.value
        t.addChild(handle(value))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    return t

handler['Yield'] = handle_Yield

# end Yield

def handle_YieldFrom(x):
    t = aterm.ATConstructor()
    t.setName("YieldFrom")
    value = x.value
    t.addChild(handle(value))

    return t

handler['YieldFrom'] = handle_YieldFrom

# end YieldFrom

def handle_Compare(x):
    t = aterm.ATConstructor()
    t.setName("Compare")
    left = x.left
    t.addChild(handle(left))

    if hasattr(x,'ops'):
        ops = x.ops
        tkid = aterm.ATList()
        for elt in ops:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    if hasattr(x,'comparators'):
        comparators = x.comparators
        tkid = aterm.ATList()
        for elt in comparators:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['Compare'] = handle_Compare

# end Compare

def handle_Call(x):
    t = aterm.ATConstructor()
    t.setName("Call")
    func = x.func
    t.addChild(handle(func))

    if hasattr(x,'args'):
        args = x.args
        tkid = aterm.ATList()
        for elt in args:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    if hasattr(x,'keywords'):
        keywords = x.keywords
        tkid = aterm.ATList()
        for elt in keywords:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    #OPTIONAL
    if hasattr(x,'starargs'):
        starargs = x.starargs
        t.addChild(handle(starargs))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    #OPTIONAL
    if hasattr(x,'kwargs'):
        kwargs = x.kwargs
        t.addChild(handle(kwargs))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    return t

handler['Call'] = handle_Call

# end Call

def handle_Num(x):
    t = aterm.ATConstructor()
    t.setName("Num")
    n = x.n
    t.addChild(handle(n))

    return t

handler['Num'] = handle_Num

# end Num

def handle_Str(x):
    t = aterm.ATConstructor()
    t.setName("Str")
    s = x.s
    t.addChild(handle(s))

    return t

handler['Str'] = handle_Str

# end Str

def handle_Bytes(x):
    t = aterm.ATConstructor()
    t.setName("Bytes")
    s = x.s
    t.addChild(handle(s))

    return t

handler['Bytes'] = handle_Bytes

# end Bytes

def handle_Ellipsis(x):
    t = aterm.ATConstructor()
    t.setName("Ellipsis")
    return t

handler['Ellipsis'] = handle_Ellipsis

# end Ellipsis

def handle_Attribute(x):
    t = aterm.ATConstructor()
    t.setName("Attribute")
    value = x.value
    t.addChild(handle(value))

    attr = x.attr
    t.addChild(handle(attr))

    ctx = x.ctx
    t.addChild(handle(ctx))

    return t

handler['Attribute'] = handle_Attribute

# end Attribute

def handle_Subscript(x):
    t = aterm.ATConstructor()
    t.setName("Subscript")
    value = x.value
    t.addChild(handle(value))

    slice = x.slice
    t.addChild(handle(slice))

    ctx = x.ctx
    t.addChild(handle(ctx))

    return t

handler['Subscript'] = handle_Subscript

# end Subscript

def handle_Starred(x):
    t = aterm.ATConstructor()
    t.setName("Starred")
    value = x.value
    t.addChild(handle(value))

    ctx = x.ctx
    t.addChild(handle(ctx))

    return t

handler['Starred'] = handle_Starred

# end Starred

def handle_Name(x):
    t = aterm.ATConstructor()
    t.setName("Name")
    id = x.id
    t.addChild(handle(id))

    ctx = x.ctx
    t.addChild(handle(ctx))

    return t

handler['Name'] = handle_Name

# end Name

def handle_List(x):
    t = aterm.ATConstructor()
    t.setName("List")
    if hasattr(x,'elts'):
        elts = x.elts
        tkid = aterm.ATList()
        for elt in elts:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    ctx = x.ctx
    t.addChild(handle(ctx))

    return t

handler['List'] = handle_List

# end List

def handle_Tuple(x):
    t = aterm.ATConstructor()
    t.setName("Tuple")
    if hasattr(x,'elts'):
        elts = x.elts
        tkid = aterm.ATList()
        for elt in elts:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    ctx = x.ctx
    t.addChild(handle(ctx))

    return t

handler['Tuple'] = handle_Tuple

# end Tuple

def handle_Load(x):
    t = aterm.ATConstructor()
    t.setName("Load")
    return t

handler['Load'] = handle_Load

# end Load

def handle_Store(x):
    t = aterm.ATConstructor()
    t.setName("Store")
    return t

handler['Store'] = handle_Store

# end Store

def handle_Del(x):
    t = aterm.ATConstructor()
    t.setName("Del")
    return t

handler['Del'] = handle_Del

# end Del

def handle_AugLoad(x):
    t = aterm.ATConstructor()
    t.setName("AugLoad")
    return t

handler['AugLoad'] = handle_AugLoad

# end AugLoad

def handle_AugStore(x):
    t = aterm.ATConstructor()
    t.setName("AugStore")
    return t

handler['AugStore'] = handle_AugStore

# end AugStore

def handle_Param(x):
    t = aterm.ATConstructor()
    t.setName("Param")
    return t

handler['Param'] = handle_Param

# end Param

def handle_Slice(x):
    t = aterm.ATConstructor()
    t.setName("Slice")
    #OPTIONAL
    if hasattr(x,'lower'):
        lower = x.lower
        t.addChild(handle(lower))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    #OPTIONAL
    if hasattr(x,'upper'):
        upper = x.upper
        t.addChild(handle(upper))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    #OPTIONAL
    if hasattr(x,'step'):
        step = x.step
        t.addChild(handle(step))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    return t

handler['Slice'] = handle_Slice

# end Slice

def handle_ExtSlice(x):
    t = aterm.ATConstructor()
    t.setName("ExtSlice")
    if hasattr(x,'dims'):
        dims = x.dims
        tkid = aterm.ATList()
        for elt in dims:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['ExtSlice'] = handle_ExtSlice

# end ExtSlice

def handle_Index(x):
    t = aterm.ATConstructor()
    t.setName("Index")
    value = x.value
    t.addChild(handle(value))

    return t

handler['Index'] = handle_Index

# end Index

def handle_And(x):
    t = aterm.ATConstructor()
    t.setName("And")
    return t

handler['And'] = handle_And

# end And

def handle_Or(x):
    t = aterm.ATConstructor()
    t.setName("Or")
    return t

handler['Or'] = handle_Or

# end Or

def handle_Add(x):
    t = aterm.ATConstructor()
    t.setName("Add")
    return t

handler['Add'] = handle_Add

# end Add

def handle_Sub(x):
    t = aterm.ATConstructor()
    t.setName("Sub")
    return t

handler['Sub'] = handle_Sub

# end Sub

def handle_Mult(x):
    t = aterm.ATConstructor()
    t.setName("Mult")
    return t

handler['Mult'] = handle_Mult

# end Mult

def handle_Div(x):
    t = aterm.ATConstructor()
    t.setName("Div")
    return t

handler['Div'] = handle_Div

# end Div

def handle_Mod(x):
    t = aterm.ATConstructor()
    t.setName("Mod")
    return t

handler['Mod'] = handle_Mod

# end Mod

def handle_Pow(x):
    t = aterm.ATConstructor()
    t.setName("Pow")
    return t

handler['Pow'] = handle_Pow

# end Pow

def handle_LShift(x):
    t = aterm.ATConstructor()
    t.setName("LShift")
    return t

handler['LShift'] = handle_LShift

# end LShift

def handle_RShift(x):
    t = aterm.ATConstructor()
    t.setName("RShift")
    return t

handler['RShift'] = handle_RShift

# end RShift

def handle_BitOr(x):
    t = aterm.ATConstructor()
    t.setName("BitOr")
    return t

handler['BitOr'] = handle_BitOr

# end BitOr

def handle_BitXor(x):
    t = aterm.ATConstructor()
    t.setName("BitXor")
    return t

handler['BitXor'] = handle_BitXor

# end BitXor

def handle_BitAnd(x):
    t = aterm.ATConstructor()
    t.setName("BitAnd")
    return t

handler['BitAnd'] = handle_BitAnd

# end BitAnd

def handle_FloorDiv(x):
    t = aterm.ATConstructor()
    t.setName("FloorDiv")
    return t

handler['FloorDiv'] = handle_FloorDiv

# end FloorDiv

def handle_Invert(x):
    t = aterm.ATConstructor()
    t.setName("Invert")
    return t

handler['Invert'] = handle_Invert

# end Invert

def handle_Not(x):
    t = aterm.ATConstructor()
    t.setName("Not")
    return t

handler['Not'] = handle_Not

# end Not

def handle_UAdd(x):
    t = aterm.ATConstructor()
    t.setName("UAdd")
    return t

handler['UAdd'] = handle_UAdd

# end UAdd

def handle_USub(x):
    t = aterm.ATConstructor()
    t.setName("USub")
    return t

handler['USub'] = handle_USub

# end USub

def handle_Eq(x):
    t = aterm.ATConstructor()
    t.setName("Eq")
    return t

handler['Eq'] = handle_Eq

# end Eq

def handle_NotEq(x):
    t = aterm.ATConstructor()
    t.setName("NotEq")
    return t

handler['NotEq'] = handle_NotEq

# end NotEq

def handle_Lt(x):
    t = aterm.ATConstructor()
    t.setName("Lt")
    return t

handler['Lt'] = handle_Lt

# end Lt

def handle_LtE(x):
    t = aterm.ATConstructor()
    t.setName("LtE")
    return t

handler['LtE'] = handle_LtE

# end LtE

def handle_Gt(x):
    t = aterm.ATConstructor()
    t.setName("Gt")
    return t

handler['Gt'] = handle_Gt

# end Gt

def handle_GtE(x):
    t = aterm.ATConstructor()
    t.setName("GtE")
    return t

handler['GtE'] = handle_GtE

# end GtE

def handle_Is(x):
    t = aterm.ATConstructor()
    t.setName("Is")
    return t

handler['Is'] = handle_Is

# end Is

def handle_IsNot(x):
    t = aterm.ATConstructor()
    t.setName("IsNot")
    return t

handler['IsNot'] = handle_IsNot

# end IsNot

def handle_In(x):
    t = aterm.ATConstructor()
    t.setName("In")
    return t

handler['In'] = handle_In

# end In

def handle_NotIn(x):
    t = aterm.ATConstructor()
    t.setName("NotIn")
    return t

handler['NotIn'] = handle_NotIn

# end NotIn

def handle_comprehension(x):
    t = aterm.ATConstructor()
    t.setName("comprehension")
    target = x.target
    t.addChild(handle(target))

    iter = x.iter
    t.addChild(handle(iter))

    if hasattr(x,'ifs'):
        ifs = x.ifs
        tkid = aterm.ATList()
        for elt in ifs:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['comprehension'] = handle_comprehension

# end comprehension

def handle_ExceptHandler(x):
    t = aterm.ATConstructor()
    t.setName("ExceptHandler")
    #OPTIONAL
    if hasattr(x,'type'):
        type = x.type
        t.addChild(handle(type))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    #OPTIONAL
    if hasattr(x,'name'):
        name = x.name
        t.addChild(handle(name))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    if hasattr(x,'body'):
        body = x.body
        tkid = aterm.ATList()
        for elt in body:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['ExceptHandler'] = handle_ExceptHandler

# end ExceptHandler

def handle_arguments(x):
    t = aterm.ATConstructor()
    t.setName("arguments")
    if hasattr(x,'args'):
        args = x.args
        tkid = aterm.ATList()
        for elt in args:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    #OPTIONAL
    if hasattr(x,'vararg'):
        vararg = x.vararg
        t.addChild(handle(vararg))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    #OPTIONAL
    if hasattr(x,'varargannotation'):
        varargannotation = x.varargannotation
        t.addChild(handle(varargannotation))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    if hasattr(x,'kwonlyargs'):
        kwonlyargs = x.kwonlyargs
        tkid = aterm.ATList()
        for elt in kwonlyargs:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    #OPTIONAL
    if hasattr(x,'kwarg'):
        kwarg = x.kwarg
        t.addChild(handle(kwarg))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    #OPTIONAL
    if hasattr(x,'kwargannotation'):
        kwargannotation = x.kwargannotation
        t.addChild(handle(kwargannotation))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    if hasattr(x,'defaults'):
        defaults = x.defaults
        tkid = aterm.ATList()
        for elt in defaults:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    if hasattr(x,'kw_defaults'):
        kw_defaults = x.kw_defaults
        tkid = aterm.ATList()
        for elt in kw_defaults:
            tkid.addChild(handle(elt))
        t.addChild(tkid)

    return t

handler['arguments'] = handle_arguments

# end arguments

def handle_arg(x):
    t = aterm.ATConstructor()
    t.setName("arg")
    arg = x.arg
    t.addChild(handle(arg))

    #OPTIONAL
    if hasattr(x,'annotation'):
        annotation = x.annotation
        t.addChild(handle(annotation))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    return t

handler['arg'] = handle_arg

# end arg

def handle_keyword(x):
    t = aterm.ATConstructor()
    t.setName("keyword")
    arg = x.arg
    t.addChild(handle(arg))

    value = x.value
    t.addChild(handle(value))

    return t

handler['keyword'] = handle_keyword

# end keyword

def handle_alias(x):
    t = aterm.ATConstructor()
    t.setName("alias")
    name = x.name
    t.addChild(handle(name))

    #OPTIONAL
    if hasattr(x,'asname'):
        asname = x.asname
        t.addChild(handle(asname))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    return t

handler['alias'] = handle_alias

# end alias

def handle_withitem(x):
    t = aterm.ATConstructor()
    t.setName("withitem")
    context_expr = x.context_expr
    t.addChild(handle(context_expr))

    #OPTIONAL
    if hasattr(x,'optional_vars'):
        optional_vars = x.optional_vars
        t.addChild(handle(optional_vars))
    else:
        tkid = aterm.ATConstant()
        tkid.setValue('None')
        t.addChild(tkid)

    return t

handler['withitem'] = handle_withitem

# end withitem



if (len(sys.argv) == 1):
    print "Specify a target file."
    sys.exit(1)


f = open(sys.argv[1],'r')
s = f.read()
f.close()


tree = ast.parse(s)

t = handle(tree)

print t.toString()

