%include {
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <map>

#include "Dynamic.h"
#include "parser.h"

using namespace std;

typedef string Key;
typedef Dynamic *Value;
typedef map<Key,Value> KeyValueMap;

}

%extra_argument   { KeyValueMap **kvm }
%token_type       { char * }
%token_destructor { free($$); }
%type kvpairs     { KeyValueMap *}
%type key         { Key *}
%type value       { Value }

%syntax_error {
  printf("Syntax error!\n");
  exit(1);
}

%parse_failure {
  fprintf(stderr,"Giving up.  Parser is lost...\n");
}

program ::= kvpairs(S) . {
  *kvm = S;
}


kvpairs(Q) ::= kvpairs(S) key(K) EQ value(V) . {
  Q = S;
  (*Q)[*K] = V;
}

kvpairs(Q) ::= . {
  Q = new KeyValueMap;
}

key(K) ::= ID(A) . {
  K = new string (A);
}

value(V) ::= ID(A) . {
  V = new DynString(A);
}

value(V) ::= NUM(A) . {
  V = new DynInt(atoi(A));
}

value(V) ::= STRING(A) . {
  V = new DynString(A);
}
