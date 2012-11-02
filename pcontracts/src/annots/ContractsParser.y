%include {
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include "ContractsParser.h"
#include "ContractClause.hpp"

using namespace std;

//%extra_argument   { ContractClause **clause }
}

%extra_argument   { ContractClause *clause }
%lang cpp
%name Contracts
%token_prefix  TOLEN_
%token_type       { char * }
%token_destructor { free($$); }
%type clause      { ContractClause * }
%type label       { char * }
%type expr        { char * }

// TBD:  Following needed?  [100]
//%stack_size 2000

%stack_overflow { 
  fprintf(stderr, "FATAL: ContractsParser stack overflow detected.\n");
  exit(1);
}

%syntax_error { 
  fprintf(stderr, "FATAL: ContractsParser syntax error detected.\n");
  exit(1);
}

%parse_failure {
  fprintf(stderr, "ERROR: ContractsParser parse failure detected.\n");
  /* *clause = NULL; */
  clause = NULL;
}

contract ::= TYPE(T) clause(A). {
  A->setType(T);
  /* *clause = A; */
  clause = A;
}

clause(C) ::= clause(I) assertions. {
  C = I;
  /* C->addAssertion(A); */
}

assertions ::= assertions assertion_expression. {
  /* TBD: What to do here? */
}

assertion_expression ::= label(L) COLON expr(E). {
  clause->addAssertion(L, E);
}

label(L) ::= LABEL(A) . {
  L = A;
}

expr(E) ::= EXPR(S) . {
  E = S;
}

clause(C) ::= . {
  C = new ContractClause();
}

