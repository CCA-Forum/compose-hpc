#include "KVAnnotationValue.h"
#include <assert.h>
#include "parser.h"
extern "C" {
#include "scanner.h"
}

Dynamic *KVAnnotationValue::lookup(string key) {
  KeyValueMap::iterator it;

  it = kvmap->find(key);
  if (it == kvmap->end()) {
    return NULL;
  } else {
    return it->second;
  }
}

KVAnnotationValue* isKVAnnotationValue( AnnotationValue *p) {
  return dynamic_cast<KVAnnotationValue *>(p);
}

KVAnnotationValue::KeyValueMap *KVAnnotationValue::parse(const string input) {

  // Lemon headers
  void *ParseAlloc(void *(*mallocProc)(size_t));
  void ParseFree(void *p,void (*freeProc)(void*));
  void Parse(void *yyp,int yymajor,char *yyminor,KeyValueMap **result);

  KeyValueMap *result;
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

void KVAnnotationValue::merge(KVAnnotationValue *other) {
  if (other == NULL)
    return;

  for (KeyValueMap::iterator it = other->kvmap->begin();
       it != other->kvmap->end();
       it++) {
    cerr << "Merging in key: " << it->first << endl;
    (*kvmap)[it->first] = it->second;
  }
}

void KVAnnotationValue::print ()
{
  typedef KeyValueMap::const_iterator CI;
  for (CI p = kvmap->begin(); p != kvmap->end(); ++p)
    cout  << p->first << ":\t" << p->second->string_value() << '\n';
}

KVAnnotationValue::KVAnnotationValue(const string s)
{
  kvmap = parse(s);
}
