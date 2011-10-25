#include <iostream>
#include "rose.h"
#include "SXAnnotationValue.h"
#include "KVAnnotationValue.h"
#include "PaulDecorate.h"

#define C_COMMENT         (1)
#define CPP_COMMENT       (2)
#define ANNOTATION_PREFIX ('%')

using namespace std;

//// Comment Visitor - code for traversing the comments ////////////////////////

class CommentVisitor : public AstSimpleProcessing {
protected:
  void virtual visit(SgNode *node);
public:
  CommentVisitor() {
  }
};

////////////////////////////////////////////////////////////////////////////////

string remove_cpp_comment_marks(const string s) {
  return s.substr(2);
}

string remove_c_comment_marks(const string s) {
  return s.substr(2,s.size() - 4);
}

bool is_annotation(const string s) {
  return s[0] == ANNOTATION_PREFIX;
}

string annotation_text(const string s) {
  return s.substr(1);
}

string annotAttributeString ("ANNOT");

void handle_comment(const string s, SgLocatedNode *node) {
  if(is_annotation(s)) {
    string ann_text = annotation_text(s);

    cerr << "Annotation=" << ann_text << endl;

    // split into the TAG and Value
      // FIXME: Make much more robust!
      //  - assumes tag exists
      //  - assumes no preceding whitespace on tag
      //  - assumes single space separator between tag & value

    string::size_type i = ann_text.find(" ");
    string tag = ann_text.substr(0,i);
    string value_text = ann_text.substr(i+1);

    // create the annotation value
      // FIXME: Add code that looks up tag in config file to see which type
      // of AnnotationValue we should create.

    KVAnnotationValue *pValue = new KVAnnotationValue (value_text);

    // create the annotation
    Annotation *pAnn = new Annotation(value_text, node, tag, pValue);

    // add the annotation to the node:
    node->addNewAttribute (annotAttributeString, pAnn);

    // tracing for now:
    cerr << "Tag: " << pAnn->getTag()
         << " ; Value: " << pAnn->getValueString()
         << endl;

  }
}

void CommentVisitor::visit(SgNode *node) {
  SgLocatedNode *locatedNode = isSgLocatedNode(node);
  if(locatedNode != NULL) {
    AttachedPreprocessingInfoType *preprocInfo;
    preprocInfo = locatedNode->getAttachedPreprocessingInfo();
    if(preprocInfo != NULL) {
      AttachedPreprocessingInfoType::iterator i;
      for(i=preprocInfo->begin(); i != preprocInfo->end(); i++) {
    	  switch ((*i)->getTypeOfDirective()) {
          case C_COMMENT: {
            string comment = remove_c_comment_marks((*i)->getString());
            handle_comment(comment,locatedNode);
            break;
          }
          case CPP_COMMENT: {
            string comment = remove_cpp_comment_marks((*i)->getString());
            handle_comment(comment,locatedNode);
            break;
          }
        }
      }
    }
  }
}

//////////////////////////

void paulDecorate (SgProject* project)
{

  CommentVisitor v;

  v.traverseInputFiles(project,preorder);    // FIXME:?
}
