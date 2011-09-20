#include "KVAnnotation.h"
#include <assert.h>
#include "parser.h"
#include "scanner.h"

// Lemon headers
void *ParseAlloc(void *(*mallocProc)(size_t));
void ParseFree(void *p,void (*freeProc)(void*));
void Parse(void *yyp,int yymajor,char *yyminor,Annotation **ann);

KVAnnotation::KVAnnotation(string s, SgNode *n) : Annotation::Annotation(s,n) {
  std::cout << "KVANNOTATION CTR: " << s << std::endl;
  // call KV parser on s, ignore n
}

void parse(const string input) {
  Annotation *result;
  yyscan_t scanner;
  void *parser;
  int yv;
  
  yylex_init(&scanner);
  parser = ParseAlloc(malloc);
  yy_scan_string(input.c_str(),scanner);
  while((yv=yylex(scanner)) != 0) {
    char *tok = yyget_extra(scanner);
    Parse(parser,yv,tok,&result);
  }
  Parse(parser,0,NULL,&result);
  ParseFree(parser,free);
  yylex_destroy(scanner);
  return result;
}

const string *KVAnnotation::lookup(string key) {
  key_value_map::iterator it;

  it = kvmap.find(key);
  if (it == kvmap.end()) {
    return NULL;
  } else {
    return &it->second;
  }
}
