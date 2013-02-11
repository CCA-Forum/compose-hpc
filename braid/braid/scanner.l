/* -*- C -*- */
%option yylineno
%option stack
%x IN_COMMENT
%x IN_DOC_COMMENT
%x MULTI1
%x MULTI2
%{
/**
 * This is a scanner for SIDL files. For performance reasons it is
 * written in C and 100% replaces the Python scanner in parser.py .
 * Please report bugs to <components@llnl.gov>.
 *
 * \authors <pre>
 *
 * Copyright (c) 2011, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory
 * Written by Adrian Prantl <adrian@llnl.gov>.
 *  
 * LLNL-CODE-473891.
 * All rights reserved.
 *  
 * This file is part of BRAID. For details, see 
 * http://compose-hpc.sourceforge.net/. 
 * Please read the COPYRIGHT file for Our Notice and
 * for the BSD License.
 *
 * </pre>
 *
 */
#include <stdlib.h>
#include <string.h>

enum Token {
          t_KEYWORD=256,
          t_IDENTIFIER, t_EXTENSION, t_VERSION_STRING,

          t_LOGICAL_AND, t_LOGICAL_OR, t_LOGICAL_XOR,

          t_LPAREN, t_RPAREN, t_LBRACE, t_RBRACE,
          t_SEMICOLON, t_COMMA, t_DOT, t_ATTRIB_BEGIN, t_ATTRIB_ID,
          t_ATTRIB_STRING, t_ATTRIB_EQ, t_ATTRIB_COMMA, t_ATTRIB_END,

          t_COMMA_COLUMN_MAJOR, t_COMMA_ROW_MAJOR, t_IDENTIFIER_COLON,

          t_ASSIGN, t_BITWISE_AND, t_BITWISE_XOR, /*t_COLON,*/ t_EQ, t_GE,
          t_GT, t_LE, t_LT, t_MINUS, t_NE, t_BITWISE_OR, t_PLUS, t_POWER,
          t_SLASH, t_STAR, t_TILDE, t_LSHIFT, t_RSHIFT,

          t_BOOLEAN_LITERAL, /*t_INTEGER_LITERAL,*/
	  t_HEX_LITERAL, t_OCTAL_LITERAL, 
          t_DECIMAL_LITERAL, t_FLOATING_POINT_LITERAL,
          t_SIMPLE_FLOATING_POINT_LITERAL, t_CHARACTER_LITERAL, t_STRING_LITERAL
};

static int yypos = 0;                  // Character count in file
static const char* yytype = NULL;      // The python token type
static char* ident = NULL;	       // Used by IDENTIFIER_COLON
static char* doc_comment = NULL;       // Used by IDENTIFIER_COLON/DOC_COMMENT
static size_t doc_comment_len = 0;     // Used by DOC_COMMENT
static size_t doc_comment_alloc = 0;   // Used by DOC_COMMENT
static const char *fn;                 // Input file name

#define update_yypos() { yypos += yyleng; }
#define ID_TOKEN       { update_yypos(); yytype = yytext; return -1; }
#define TOKEN(TOKEN)   { update_yypos(); yytype = #TOKEN; return t_##TOKEN; }
#define max(X,Y)       ((X) > (Y) ? (X) : (Y))
#define new_doxy()     { if (doc_comment) {doc_comment[0] = 0; doc_comment_len = 0;} }
// Concat yytext+offset to the (growing) doc_comment buffer
static void concat_doxy(offset)                                                
{									   
  update_yypos();							   
  if (!doc_comment) {							   
    doc_comment_alloc = 1024;						   
    doc_comment = malloc(doc_comment_alloc);				   
    doc_comment_len = 0;						   
    doc_comment[0] = 0;							   
  }									   
  if (yyleng-offset + doc_comment_len >= doc_comment_alloc) {		   
    doc_comment_alloc = max(doc_comment_alloc*2, 1024);			   
    doc_comment = realloc(doc_comment, doc_comment_alloc); 
  }									   
  strncat(doc_comment, yytext+offset, doc_comment_alloc-doc_comment_len);		   
  doc_comment_len += yyleng - offset;					   
}						      

%}

DECIMAL_LITERAL               [1-9][0-9]*
HEX_LITERAL                   0[xX][0-9a-fA-F]+
OCTAL_LITERAL                 0[0-7]*
SIMPLE_FLOATING_POINT_LITERAL [0-9]+\.[0-9]+
EXPONENT                      [eE][\+-]?[0-9]+
CHARACTER_LITERAL             '(~['\\\n\r]|(\\[ntbrf\\'"]|([0-7][0-7]?)|([0-3][0-7][0-7]))'
STRING_LITERAL                "(~["\\\n\r]|(\\[ntbrf\\'"]|([0-7][0-7]?)|([0-3][0-7][0-7]))"
IDENTIFIER                    [a-zA-Z][a-zA-Z_0-9]*

%%
 /*defeat emacs syntax highlighting: "'*/


<INITIAL,MULTI1,MULTI2>{
  [ \t\f] update_yypos(); // White Space
  [\r\n]  update_yypos();

  "///".* new_doxy(); concat_doxy(3);                                // C++ Doxygen comment
  "//".*  update_yypos();                                            // C++ comment
  "/**"   new_doxy(); update_yypos(); yy_push_state(IN_DOC_COMMENT); // C Doxygen comment
  "/*"    update_yypos(); yy_push_state(IN_COMMENT);                 // C comment
}
<IN_COMMENT>{
  "*/"        update_yypos(); yy_pop_state();
  [^*\n]+     update_yypos(); // eat comment in chunks
  "*"         update_yypos(); // eat the lone star
  [\r\n]      update_yypos();
}
<IN_DOC_COMMENT>{
  "*/"        update_yypos(); yy_pop_state();
  [^*\n]+     concat_doxy(0); // this adds the contents of yytext to doc_comment
  "*"         concat_doxy(0);
  [\r\n]+[ \t]*"*"[^/] {      // ignore '*' at the beginning of a line
    yyless(yyleng-1);
    yytext[yyleng-1] = ' '; concat_doxy(0); 
  }
  [\r\n]      concat_doxy(0);
}

  /* TODO  <"/**" ~["/"] > { input_stream.backup(1); } : IN_DOC_COMMENT */

  /* Keywords */
void|array|rarray|bool|char|dcomplex|double|fcomplex|float|int|long|opaque|string |
class|enum|struct|interface|abstract|copy |
else|ensure|extends|final|from|iff|implements|implements-all|implies|import|in |
inout|invariant|is|local|mod|not|null|nonblocking|oneway|order|out|package |
pure|rem|require|result |
static|then|throws|version {
  char *c;
  for (c = yytext; c < yytext+yyleng; c++) {
    // parser expects upper case for terminal symbols
    if (*c == '-')
      *c = '_';
    else
      *c = toupper(*c);
  }
  ID_TOKEN
}

and TOKEN(LOGICAL_AND)
or  TOKEN(LOGICAL_OR)
xor TOKEN(LOGICAL_XOR)
false|true ID_TOKEN


"["[a-zA-Z0-9_]+"]"      TOKEN(EXTENSION)
[0-9]\.[0-9]+(\.[0-9]+)+ TOKEN(VERSION_STRING)

  /* Work around the fact that we need a lookahead of 2 for the
     grammar at some points */

<INITIAL>{
  ","           BEGIN(MULTI1); update_yypos();
}
<MULTI1>{
  row-major     BEGIN(INITIAL);            TOKEN(COMMA_ROW_MAJOR);
  column-major  BEGIN(INITIAL);            TOKEN(COMMA_COLUMN_MAJOR);
  .             BEGIN(INITIAL); yyless(0); TOKEN(COMMA);
}

<INITIAL>{
  {IDENTIFIER}  BEGIN(MULTI2); update_yypos(); ident = strdup(yytext);
}
<MULTI2>{
  ":"           BEGIN(INITIAL);            TOKEN(IDENTIFIER_COLON)
   .            BEGIN(INITIAL); yyless(0); TOKEN(IDENTIFIER)
}

  /* separators */
"("   TOKEN(LPAREN)
")"   TOKEN(RPAREN)
"{"   TOKEN(LBRACE)
"}"   TOKEN(RBRACE)
";"   TOKEN(SEMICOLON)
"."   TOKEN(DOT)

  /* operators */
"="   TOKEN(ASSIGN)
"&"   TOKEN(BITWISE_AND)
"^"   TOKEN(BITWISE_XOR)
"=="  TOKEN(EQ)
">="  TOKEN(GE)
">"   TOKEN(GT)
"<="  TOKEN(LE)
"<"   TOKEN(LT)
"-"   TOKEN(MINUS)
"!="  TOKEN(NE)
"|"   TOKEN(BITWISE_OR)
"+"   TOKEN(PLUS)
"**"  TOKEN(POWER)
"/"   TOKEN(SLASH)
"*"   TOKEN(STAR)
"~"   TOKEN(TILDE)
"<<<" TOKEN(LSHIFT)
">>>" TOKEN(RSHIFT)

  /* literals */
{DECIMAL_LITERAL}[lL]?          TOKEN(DECIMAL_LITERAL)
{HEX_LITERAL}[lL]?              TOKEN(HEX_LITERAL)
{OCTAL_LITERAL}[lL]?            TOKEN(OCTAL_LITERAL)
{SIMPLE_FLOATING_POINT_LITERAL} TOKEN(SIMPLE_FLOATING_POINT_LITERAL)

{SIMPLE_FLOATING_POINT_LITERAL}{EXPONENT}[fFdD]? |
{SIMPLE_FLOATING_POINT_LITERAL}({EXPONENT})?[fFdD] |
[0-9]+\.{EXPONENT}?[fFdD]? |
\.[0-9]+{EXPONENT}?[fFdD]? |
[0-9]+{EXPONENT}[fFdD]? |
[0-9]+{EXPONENT}?[fFdD] {
        TOKEN(FLOATING_POINT_LITERAL)
}

<<EOF>>   return 0;

. {
    fprintf(stderr, "**ERROR: %s:%d:\n  unexpected token '%s'.\n",
            fn, yylineno, yytext);
    exit(1);
}
%%

#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif
#include <Python.h>
#include <stdio.h>

static PyObject* token_module = NULL;
static PyObject* token_dict   = NULL;
static PyObject* token_init   = NULL;

static PyObject*
scanner_input(PyObject *self, PyObject *args)
{
  if (!PyArg_ParseTuple(args, "s", &fn))
    return Py_None;
  yyin = fopen(fn, "r");
  return Py_BuildValue("i", 1);
}

static void crash(const char* msg) {
  fprintf(stderr, "** FATAL ERROR: %s\n", msg);
  exit(1);
}

static PyObject*
scanner_token(PyObject *self, PyObject *args)
{
  static PyObject* r;
  int type = yylex();
  //while (yylex()) { fprintf(stdout, "token:'%s' :: %s\n", yytext, yytype); fflush(stdout);} exit(1);

  // EOF
  if (type == 0 && token_init) {
    // Free references to Token
    // FIXME: this segfaults upon restarting: Py_DECREF(token_module); 
    // FIXME: this segfaults upon restarting: Py_DECREF(token_dict); 
    // FIXME: this triggers a segfault: Py_DECREF(token_init);
    fclose(yyin);
    free(doc_comment);
    doc_comment = NULL;
    doc_comment_alloc = 0;
    doc_comment_len = 0;
    yypos = 0;
    return Py_None;
  }

  // Initialization
  if (token_init == NULL) {
     // Load the Token module
     token_module = PyImport_Import(PyString_FromString("sidl_token"));
     if (!token_module) { 
       crash("Could not open sidl_token.py[oc]"); 
     }
     token_dict = PyModule_GetDict(token_module);
     if (!token_dict) { 
       crash("Could not open dictionary in sidl_token module"); 
     }
     token_init = PyDict_GetItemString(token_dict, "Token");
     if (!token_init) { 
       crash("Could not locate definition of Token()"); 
     }
  }
  //fprintf(stderr, "(%d, %d), type: %s, token: `%s'\n", yylineno, yypos, yytype, yytext); fflush(stderr);
  
  // Build python return value
  r = PyObject_CallFunction(token_init, "(ssii)", yytype, ident?ident:yytext,
  			    yylineno, yypos);
  free(ident);
  ident = NULL;

  return r;
}

static PyObject*
last_doc_comment(PyObject *self, PyObject *args)
{
  static PyObject* r;
  //fprintf(stderr, "doc: `%s, %d'\n", doc_comment, doc_comment_len); fflush(stderr);
  r = Py_BuildValue("s#", doc_comment?doc_comment:"", doc_comment_len);
  new_doxy(); // reset the docstring
  return r;
}

static PyMethodDef ScannerMethods[] = {
    {"input",  scanner_input, METH_VARARGS,
     "set the input stream."},
    {"token",  scanner_token, METH_VARARGS,
     "return a new token from the input stream."},
    {"last_doc_comment",  last_doc_comment, METH_VARARGS,
     "return the last doc comment read from the input stream."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initscanner(void)
{
    (void) Py_InitModule("scanner", ScannerMethods);
}
