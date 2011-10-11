import re, sys
from parse_tree import *

def error(msg):
    print '**ERROR', msg
    print 'in line', line_number
    exit(1)

rules = []
nonterminals = set()
line_number = 0
state = 'base'
f = open(sys.argv[1])
for line in f.readlines():
    line_number += 1
    if re.match(r'^\s*?(#.*)?$', line): # Comment
        continue

    sig = re.match(r'^%action arguments: \((.*)\)$', line)
    if sig: # Comment
        signature = sig.group(1) 
        continue

    if state == 'in_rule':
        if re.match(r'^\s.*$', line):
            print line, # no extra newline
            have_rule_body = True
        else:
            if not have_rule_body:
                print '    pass'
            print

            state = 'base'

    if state == 'base':
        if not re.match(r'\s', line[0]):
            # new rule

            # we are using the following grammar:
            # target @lang -> src @lang? : cost(c)
            #      code
            m = re.match('(.*)@(.+)->(.*?)(@(.+))?:cost\((\d+)\)',
                         re.sub('\s','', line).replace('.','_'))
            if not m: error('Bad rule syntax\n'+line)

            def chk(match, n, errmsg):
                return match.group(n) if match.group(n) \
                    else error('Syntax Error in '+errmsg)

            # parse the components of the rule declaration
            target      = chk(m, 1, 'target')
            target_lang = chk(m, 2, 'target lang')
            src         = parse_tree(chk(m, 3, 'src'))
            src_lang    = m.group(5) if m.group(5) else target_lang
            cost        = int(chk(m, 6, 'cost'))

            # add the rule to our internal bookkeeping
            action = re.sub(r"[ '\(\)]",'', str(src).replace(',','_'))+'_to_'+target
            nonterminals.add(target)
            rules.append((target, src, cost, action))

            # print the rule action
            print '#', line,
            print 'def %s(%s):'%(action, signature)
            state = 'in_rule'
            have_rule_body = False

        else: print '##ERR?'

print
print 'rules = ['
for target, src, cost, action in rules:
    print '    (%r, %r, %r, %s),' % (target, src, cost, action)
print '  ]'
print 'nonterminals = ', repr(nonterminals)
print
print 'def no_action(*args): pass'
print '''

print 'rules = ', repr(rules)
from parse_tree import *

def label(tree):
    """
    Find a cost-minimal cover of the tree \c tree using the the rules
    defined in the global variable \c rules.
    """
    return label1(parse_tree(tree.replace('.','_')))

def label1(node):
    if isinstance(node, tuple):
        arity = len(node)
        child_labels = map(label1,node[1:])
        my_labels = dict()
        functor = node[0]
    else:
        # FIXME (performance) replace this with a hardcoded array
        my_labels = { node: ((node, '<terminal>', 0, no_action), 0) }
        arity = 0
        functor = node

    def current_cost(target):
        for (t, src, _, action), cost in my_labels.values():
            if target == t:
                print "current_cost(%s) = %s"%(target,cost);
                return cost
        return 2**16

    if functor in nonterminals:
        print 'node %r is a non-terminal symbol'%functor
        exit(1)

    print "label1(%s):"%functor
    #print "my_labels: ", my_labels

    visited = set()
    fixpoint = False
    while not fixpoint:
        fixpoint = True
        for r in rules:
            target, src, cost, action = r

            #print 'src =', src
            #import pdb; pdb.set_trace()
            if arity and not (isinstance(src, tuple) 
                              and len(src) == arity 
                              and src[0] == functor):
                continue # not compatible

            if arity == 0:
                try:    _, basecost = my_labels[src]
                except: continue # rule does not match
                
            for i in range(1, arity):
                try:    _, basecost = child_labels[src[i]]
                except: continue # rule does not match

            # decide whether it pays off to add this rule
            if cost < current_cost(target) and target not in visited:
            #if src not in visited:
                visited.add(src)
                my_labels[target] = (r, cost)
                print '    my_labels[',target,'] = ',(r, cost)
                fixpoint = False

    # debug output
    for r, cost in my_labels.values():
        print '   ', r, ':', cost

    if isinstance(node, tuple):
        return my_labels
    else: return my_labels

def reducetree(label, target, *args):
    print "cost-optimal cover:"
    return reducetree1(label, target.replace('.','_'), *args)

def reducetree1(label, target, *args):
    """
    Reduce a tree of labels (as generated by \c label() ) to \c target
    and execute all the code generation action associated with the
    labels as side effects.
    """
    if isinstance(label, tuple):
        my_labels = label[0]
    else:
        my_labels = label

    success = False

    while target in my_labels:
        r, cost = my_labels[target]
        del my_labels[target]

        _, target, _, action = r
        print r, cost
        action(*args)
        success = True

    if not success:
        print "**ERROR: no cover found!"

    if isinstance(label, tuple):
        reducetree1(label[1], target, *args)


try:
    reducetree(label(('chpl.Char')), 'ior.str', [], set())
    print
    reducetree(label('upcast(ior.object)'), 'ior.baseobject', [], set())
except:
    # Invoke the post-mortem debugger
    import pdb, sys
    print sys.exc_info()
    pdb.post_mortem()

'''
