import re, sys


def error(msg):
    print '**ERROR', msg
    print 'in line', line_number
    exit(1)

rules = []
all_symbols = set()
nonterminals = set()
line_number = 0
state = 'base'
f = open(sys.argv[1])
for line in f.readlines():
    line_number += 1
    if re.match(r'^\s*?(#.*)?$', line): # Comment
        continue

    if state == 'in_rule':
        if re.match(r'^\s.*$', line):
            pass #print line
        else:
            state = 'base'

    if state == 'base':
        if not re.match(r'\s', line[0]):
            # new rule
            m = re.match('(.*)@(.+)->(.*?)(@(.+))?:cost\((\d+)\)',
                         re.sub('\s','', line))
            if not m: error('Bad rule syntax\n'+line)

            def chk(match, n, errmsg):
                return match.group(n) if match.group(n) \
                    else error('Syntax Error in '+errmsg)

            target = chk(m, 1, 'target')
            target_lang = chk(m, 2, 'target lang')
            src = chk(m, 3, 'src')
            src_lang = m.group(5) if m.group(5) else target_lang
            cost = int(chk(m, 6, 'cost'))

            #print '# target =', target, 'target_lang =', target_lang, \
            #    'src =', src, 'src_lang =', src_lang, 'cost = ', cost

            # # add rule to our reverse-sorted dict
            # if src not in srcs:
            #     srcs[src] = []
            # srcs[src] = [(target, src)]+srcs[src]
            all_symbols.add(src)
            all_symbols.add(target)
            nonterminals.add(target)
            rules.append((target, src, cost))

            #print "def "+target+'_to_'+src+'(convs, optional):'
            state = 'in_rule'

#print 'rules = ', repr(rules)

def label(node):
    if isinstance(node, tuple):
        c0_labels = label(node[1])
        my_labels = c0_labels
        thisnode = node[0]
    else:
        # FIXME replace this with a hardcoded array
        my_labels = { node: ((node, '<terminal>', 0), 0) }
        thisnode = node

    def current_cost(target):
        for (t, src, _), cost in my_labels.values():
            if target == t:
                #print "current_cost(%s) = %s"%(target,cost);
                return cost
        return 2**16

    if thisnode in nonterminals:
        error('node %r is a non-terminal symbol'%thisnode)

    print "label(%s):"%thisnode
    #print "my_labels: ", my_labels

    visited = set()
    fix = False
    while not fix:
        fix = True
        for r in rules:
            target, src, cost = r
            try:
                _, basecost = my_labels[src]
                # does it pay off to add this rule?
                if cost < current_cost(target) and target not in visited:
                    visited.add(src)
                    my_labels[target] = (r, cost)
                    print '    my_labels[',target,'] = ',(r, cost)
                    fix = False
            except: pass

    # debug output
    for r, cost in my_labels.values():
        print '   ', r, ':', cost

    if isinstance(node, tuple):
        return my_labels, c0_labels
    else: return my_labels

def reducetree(label, target):
    if isinstance(label, tuple):
        my_labels = label[0]
    else:
        my_labels = label

    # cheapest = None
    # min_cost = 2**16
    # for r, cost in my_labels.values():
    #     if cost < min_cost:
    #         cheapest = r
    # if cheapest:
    #     _, next_target, _ = cheapest
    #     print repr(cheapest)
    while target in my_labels:
        r, cost = my_labels[target]
        del my_labels[target]

        _, target, _ = r
        print r, cost

    if isinstance(label, tuple):
        reducetree(label[1], target)



try:
    reducetree(label(('chpl.Char')), 'ior.str')
except:
    # Invoke the post-mortem debugger
    import pdb, sys
    print sys.exc_info()
    pdb.post_mortem()
