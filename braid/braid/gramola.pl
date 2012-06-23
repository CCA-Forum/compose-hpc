#!/usr/bin/swipl -q -O -t main -s
% -*- prolog -*-, not perl
/**
 * @package gramola
 *
 * Python grammar code generator
 *
 * GRAMOLA is a code generator for BRAID intermediate representation grammars.
 * <pre>
 *   `I'd rather write a program to write programs to write programs
 *    than write a program to write programs.'
 * </pre>
 *
 * Input: a grammar specification
 *
 * Output: Python code to create and typecheck expressions of that grammar
 *
 * Usage:
 * <pre>
 *   egrep '^%[^%]' $< | sed -e 's/^% %%/##/g' -e 's/^%//g' >$@
 *   swipl -f gramola.pl -t main -q <grammar_def.pl >>ir_def.py
 * </pre>
 */

% Before doing anything else, set up handling to halt if warnings or errors
% are encountered.
:- dynamic prolog_reported_problems/0.
% If the Prolog compiler encounters warnings or errors, record this fact in
% a global flag. We let message_hook fail because that signals Prolog to
% format output as usual (otherwise, we would have to worry about formatting
% error messages).
user:message_hook(_Term, warning, _Lines) :-
    assert(prolog_reported_problems), !, fail.
user:message_hook(_Term, error, _Lines) :-
    assert(prolog_reported_problems), !, fail.


:- use_module(library(ordsets)).

main(_) :-
    % Enable the graphical debugger, if it is installed.
    (   predicate_property(guitracer, _)
    ->  guitracer
    ;   true ),
	
    format('~n~n# Automatically generated by GRAMOLA.~n'),
    format('#     ### ### #### ### ###~n'),
    format('#     ### DO NOT EDIT! ###~n'),
    format('#     ### ### #### ### ###~n'),
    format('import types as PythonTypes~n'),
    prompt(_, ''),
    (	read_term(Term, [variable_names(Varnames),
			 singletons(warning),
			 syntax_errors(fail)])
    ;	format(user_error, '**Syntax error.~n', []), halt(1)
    ),
    normalize(Term, Docstrings, Varnames1),
    copy_term(Docstrings, CyclicGrammar),
    copy_term(Docstrings-Varnames, Gacc-Vacc),

    format('~n~n## Token definitions~n~n'),
    copy_term(Docstrings, G),
    gather_tokens(G, Tokens),
    maplist(tokendef, Tokens), !,

    format('~n~n## Constructor definitions~n~n'),
    % bind free variables in the docstrings to symbol names
    maplist(docref, Varnames),
    maplist(docref, Varnames1), !,
    % FIXME: use try, catch instead
    (	maplist(unify, CyclicGrammar)
    ;	format(user_error,
	       '**ERROR: did you use the same symbol twice on the LHS?~n', []),
	halt(1)), !,
    maplist(rhs_of, CyclicGrammar, CycGrammarRHS), !,
    maplist(constructor, CycGrammarRHS, Docstrings),

    % generate the accessor functions
    format('~n~n## Accessor functions~n~n'),
    maplist(lownames, Vacc),
    maplist(rhs_of, Gacc, GaccRHS), !,
    maplist(accessor, GaccRHS, Docstrings),

    format('~n~n## instance checks~n~n'),
    maplist(instanceof, GaccRHS, Docstrings).

%    format('~n~n## Traversal classes~n~n'),
%    format('def DepthFirstPostOrderTraversal():~n'),
%    format('    """~n'),
%    format('    Inherit from this class for a Depth-first post-order traversal.~n'),
%    format('    """~n'),
    
main(_) :-
    format(user_error, 'Internal error. Please complain to <adrian@llnl.gov>.~n', []).

% all reserved words in the Python language
python_reserved_word(A) :-
    memberchk(A, ['False','True','None','NotImplemented','Ellipsis',
		   and,as,assert,break,class,continue,def,del,elif,
		   else,except,exec,finally,for,from,global,if,import,
		   in,is,lambda,not,or,pass,print,raise,return,try,
		   while,with,yield]).

safe_atom(Atom, Safe) :-
    (	python_reserved_word(Atom)
    ->	atom_concat(Atom, '_', Safe)
    ;	Atom = Safe).

rhs_of(_=B, B).
unify(A=B) :- A = B, !.
unify(Fail) :- format(user_error, '**Failed in rule `~w\'~n', [Fail]), fail.
docref(Name=Var) :-
    safe_atom(Name, NameS),
    atom_concat('\\c ', NameS, Var).
lownames(Name=Var) :-
    lowercase_atom(Name, NameL),
    safe_atom(NameL, Var).

% ignore python builtins
tokendef('STR').
tokendef('FLOAT').
tokendef('INT').
tokendef(Token) :-
    safe_atom(Token, TokenS),
    format('~a = \'~a\'~n', [TokenS, Token]).

% recursively collect all functors from the grammar
gather_tokens([], []).
gather_tokens(V, []) :- var(V).
gather_tokens(_=B, Ts) :- gather_tokens(B, Ts).
gather_tokens([A|As], Ts) :-
    gather_tokens(A, TA),
    gather_tokens(As, TAs),
    ord_union(TA, TAs, Ts).
gather_tokens(A|B, Ts) :-
    gather_tokens(A, TAs),
    gather_tokens(B, TBs),
    ord_union(TAs, TBs, Ts).
gather_tokens(A, [A]) :- atom(A).
gather_tokens(A, Ts1) :-
    A =.. [Name|As],
    gather_tokens(As, Ts),
    ord_union([Name], Ts, Ts1).

% ----------------------------------------------------------------------
%%normalize/2
% Rewrite the grammar such that each nested complex term on the RHS is
% replaced by a Variable plus an additional rule defining that
% variable.
normalize([], [], []).
normalize([A=B|Rules], NormalizedRules, Varnames) :-
    normalize_rhs(B, Bn, AdditionalRules, Varnames1, yes),
    normalize(Rules, RulesN, Varnames2),
    ord_add_element(AdditionalRules, A=Bn, AdditionalRules1),
    ord_union(RulesN, AdditionalRules1, NormalizedRules),
    append(Varnames1, Varnames2, Varnames). 
    
normalize_rhs(Var, Var, [], [], _) :- var(Var), !.
normalize_rhs(Atom, Atom, [], [], _) :- atom(Atom), !.
normalize_rhs([List], [List1], Rules, Varnames, _) :- !,
    normalize_rhs(List, List1, Rules, Varnames, no).
normalize_rhs(A|B, An|Bn, Rules, Varnames, _) :- !,
    normalize_rhs(A, An, RulesA, VarnamesA, no),
    normalize_rhs(B, Bn, RulesB, VarnamesB, no),
    ord_union(RulesA, RulesB, Rules),
    append(VarnamesA, VarnamesB, Varnames).
% we recurse only into the outermost complex term
normalize_rhs(Complex, ComplexN, Rules, Varnames, yes) :- !,
    Complex =.. [Type|Args],
    normalize_list(Args, ArgsN, Rules, Varnames),
    ComplexN =.. [Type|ArgsN].
% the interesting case
normalize_rhs(Complex, Var, [Var=Complex], [Name=Var], _) :-
    functor(Complex, Type, _),
    uppercase_atom(Type, Name).

normalize_list([], [], [], []).
normalize_list([A|As], [An|Ans], Rules, Varnames) :-
    normalize_rhs(A, An, RulesA, VarnamesA, no),
    normalize_list(As, Ans, RulesAs, VarnamesAns),
    ord_union(RulesA, RulesAs, Rules),
    append(VarnamesA, VarnamesAns, Varnames).
% ----------------------------------------------------------------------


% helper for alternatives
validation1(Arg, Var, Indent) :- !,
    format('~ael', [Indent]),
    (	Arg = (A|B)
    ->	type_check(A, Var, Indent),
	validation1(B, Var, Indent)
    ;	type_check(Arg, Var, Indent)
    ).

%% validation/2: output validation code
% (=a type constructor) for a given grammar node
% alternatives
validation(Arg, Var, Indent) :- !,
    %A =.. [Type|Args],
    format(Indent),
    (	Arg = (A|B)
    ->	type_check(A, Var, Indent),
	validation1(B, Var, Indent)
    ;	type_check(Arg, Var, Indent)
    ),
    format('~aelse:~n', [Indent]),
    format('~a    print f.__name__+"():\\n    \\"\\"\\"%s\\"\\"\\"\\n" \
	%f.__doc__.replace("\\\\n","\\n")\
                  .replace("\\return","Returns")\
                  .replace("\\\\c ","")~n',
	  [Indent]),
    format('~a    print "**GRAMMAR ERROR in argument \
	    ~a = %s"%repr(~a)~n', [Indent, Var, Var]),
    format('~a    print "  Most likely you now want to enter \\"up<enter>l<enter>\\"\\n \
	   into the debugger to see what happened.\\n"~n', [Indent]),	    
    format('~a    raise Exception("Grammar Error")~n', [Indent]).

% builtin
builtin_name('STR', 'PythonTypes.StringType').
builtin_name('INT', 'PythonTypes.IntType').
builtin_name('FLOAT', 'PythonTypes.FloatType').
type_check(A, Var, Indent) :-
    builtin_name(A, BuiltinType),
    format('if isinstance(~a, ~a):~n', [Var, BuiltinType]),
    format('~a    pass~n', [Indent]).

% atom
type_check(A, Var, Indent) :-
    atom(A),
    safe_atom(A, AS),
    format('if ~a == ~a:~n', [Var, AS]),
    format('~a    pass~n', [Indent]).

% list: accept lists and tuples
type_check([A], Var, Indent) :-
    format('if isinstance(~a, list) or isinstance(~a, tuple):~n', [Var, Var]),
    format('~a    for a in ~a:~n', [Indent, Var]),
    atom_concat(Indent, '        ', Indent1),
    validation(A, 'a', Indent1).

% alternatives
type_check(A|B, Var, Indent) :-
    type_check(A, Var, Indent),
    validation1(B, Var, Indent).

% complex term
type_check(A, Var, Indent) :-
    A =.. [Type|_],
    safe_atom(Type, TypeS),
    format('if isinstance(~a, tuple) and ~a[0] == ~a:~n', [Var, Var, TypeS]),
    format('~a    pass~n', [Indent]).

%% validations/2
% iterate over all elements of a complex term and print validation code
validations(Args, Var) :-
    length(Args, L),
    format('    if len(~a) <> ~d:~n', [Var, L]),
    format('        print "**GRAMMAR ERROR: expected ~d arguments for a", f.__name__~n', [L]),
    format('        print "Most likely you want to enter \\"up<enter>l<enter>\\" \
	  now to see what happened."~n'),
    format('        raise Exception("Grammar Error")~n'),
    validations(Args, Var, 0).
validations([], _,
	    _).
validations([Arg|Args], Var, I) :-
    format(atom(Var1), '~a[~d]', [Var, I]),
    validation(Arg, Var1, '    '),
    %check_arg(Arg, Var, I),
    I1 is I+1,
    validations(Args, Var, I1).

%% uppercase_atom/2 convert 'abc' to 'Abc'
uppercase_atom(A, A1) :-
    atom_chars(A, [C|Cs]),
    char_type(C1, to_upper(C)),
    atom_chars(A1, [C1|Cs]).

%% lowercase_atom/2 convert 'Abc' to 'abc'
lowercase_atom(A, A1) :-
    atom_chars(A, [C|Cs]),
    char_type(C1, to_lower(C)),
    atom_chars(A1, [C1|Cs]).


%% constructor/2: output a constructor for a given grammar node
constructor([], _).
constructor(_A|_B, Doc)    :- format('# skipping ~w~n', [Doc]).
constructor([_|_], Doc) :- format('# skipping ~w~n', [Doc]).
constructor(Atom, _) :-
    atom(Atom),
    uppercase_atom(Atom, Def),
    safe_atom(Def, Def1),
    format('def ~a():~n    return ~a~n', [Def1, Atom]).

constructor(Term, Docstring) :-
    ground(Term),		% sanity check
    Term =.. [Type|[Arg|Args]],
    pretty(Docstring, Doc),
    uppercase_atom(Type, Def),
    safe_atom(Def, Def1),
    format('def ~a(*args):~n', [Def1]),
    format('    """~n'),
    format('    Construct a "~a" node. Valid arguments are ~n    ~w~n', [Type, Doc]),
    format('    """~n'),
    format('    f = ~a~n', [Def1]),
    validations([Arg|Args], 'args'),
    format('    return tuple([\'~a\']+list(args))~n~n', [Type]).

constructor(Error, _) :-
    format(user_error, '**ERROR: In ~w~n', [Error]),
    (	ground(Error),
	format(user_error, 'Internal error, spawning debugger.~n', []),
	gtrace
    ;	format(user_error, 'Most probably missing a definition in above term.~n', []),
	format(user_error, 'Until better error recovery is implemented, ~n', []),
	format(user_error, 'please look for symbols like `_G123\' in the output~n', []),
    	format(user_error, 'above and compare that to the grammar definition~n', []),
	format(user_error, 'to identify the offending term.~n', [])
    ),
    halt(1).

% ----------------------------------------------------------------------

%% accessor/2: output an accessor function for a given grammar node
accessor([], _).
accessor(_A|_B, Doc) :- format('# skipping ~w~n', [Doc]).
accessor([_|_], Doc) :- format('# skipping ~w~n', [Doc]).
accessor(Atom, Doc) :- atom(Atom), format('# skipping ~w~n', [Doc]).

accessor(Term, _) :-
    ground(Term),		% sanity check
    Term =.. [Type|Args],
    accessor1(Type, 1, Args).

accessor(Error, _) :-
    format(user_error, '**ERROR: In ~w~n', [Error]),
    halt(1).

%% get all the different names an argument may have via backtracking
name_of(Arg, ArgName) :- name_of(Arg, ArgName, '').
name_of([Arg], ArgName, _) :- !, name_of(Arg, ArgName, 's').
name_of(A|_, ArgName, S) :- name_of(A, ArgName, S).
name_of(_|B, ArgName, S) :- !, name_of(B, ArgName, S).
name_of(Arg, ArgName, S) :-
	functor(Arg, F, _),
	sub_atom(F, _, 1, 0, Last),
	( Last = 's' % append plural `s' only of last char is not 's'
	-> ArgName = F
	; atom_concat(F, S, ArgName)).

accessor1(_, _, []).
accessor1(Type, N, [Arg|Args]) :-    
    ( name_of(Arg, ArgName),
      format('def ~a_~a(arg):~n', [Type, ArgName]),
      format('    """~n'),
      format('    Accessor function.~n'),
      format('    \\return the "~a" member of a "~a" node.~n', [ArgName, Type]),
      format('    """~n'),
      format('    if not isinstance(arg, tuple):~n', []),
      format('        raise Exception("Grammar Error")~n'),
      format('    elif arg[0] <> \'~a\':~n', [Type]),
      format('        raise Exception("Grammar Error")~n'),
      format('    else: return arg[~d]~n~n~n', [N]),
      fail % loop until we gone through all alternatives A|B
    )
    ; (
       N1 is N+1,
       accessor1(Type, N1, Args)
      ).

% ----------------------------------------------------------------------

%% instanceof/2: output an instanceof function for a given grammar node
instanceof([], _).
instanceof(_A|_B, Doc) :- format('# skipping ~w~n', [Doc]).
instanceof([_|_], Doc) :- format('# skipping ~w~n', [Doc]).
instanceof(Atom, Doc) :- atom(Atom), format('# skipping ~w~n', [Doc]).

instanceof(Term, _) :-	       % functors: this is the intersting case
    ground(Term),	       % sanity check
    Term =.. [Type|_],
    format('def is_~a(arg):~n', [Type]),
    format('    """~n'),
    format('    instanceof-like function.~n'),
    format('    \\return \\c True if the argument is a "~a" node.~n', [Type]),
    format('    """~n'),
    format('    if not isinstance(arg, tuple):~n', []),
    format('        return False~n'),
    format('    return arg[0] == \'~a\'~n~n', [Type]).

instanceof(Error, _) :-
    format(user_error, '**ERROR: In ~w~n', [Error]),
    halt(1).

% ----------------------------------------------------------------------

% pretty-print grammar for the docstring
pretty(Atom, PrettyAtom) :-
    atom(Atom),
    uppercase_atom(Atom, A1), safe_atom(A1,A2),
    format(atom(PrettyAtom), '~a()', [A2]).
pretty([List], PrettyList1) :-
    pretty(List, PrettyList),
    format(atom(PrettyList1), '[~w]', [PrettyList]).
pretty(A|B, PrettyTerm) :-
    pretty(A, PrettyA),
    pretty(B, PrettyB),
    format(atom(PrettyTerm), '~w~n    |~w', [PrettyA, PrettyB]).
pretty(A=B, Rule) :-
    B =.. [_|Args],
    sub_atom(A, 3, _, 0, ShortA),
    maplist(pretty, Args, PrettyArgs),
    atomic_list_concat(PrettyArgs, ', ', PrettyArgs1),
    format(atom(Rule), '(~w)~n    \\return (\\c "~a", ~w)',
	   [PrettyArgs1, ShortA, PrettyArgs1]).
pretty(Complex, Rule) :-
    Complex =.. [Type|Args],
    maplist(pretty, Args, PrettyArgs),
    atomic_list_concat(PrettyArgs, ', ', PrettyArgs1),
    format(atom(Rule), '(~w)~n    \\return (\\c "~a", ~w)',
	   [PrettyArgs1, Type, PrettyArgs1]).

% ----------------------------------------------------------------------

% Finish error handling (see top of source file) by halting with an error
% condition of Prolog generated any warnings or errors.
:- (prolog_reported_problems -> halt(1) ; true).
