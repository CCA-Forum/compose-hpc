#include <iostream>
#include "rose.h"
#include "SXAnnotationValue.h"
#include "KVAnnotationValue.h"
#include "PaulDecorate.h"
#include "PaulConfReader.h"

#define C_COMMENT         (1)
#define CPP_COMMENT       (2)
#define ANNOTATION_PREFIX ('%')

using namespace std;

//// Comment Visitor - code for traversing the comments ////////////////////////

class CommentVisitor : public AstSimpleProcessing {
private:
  paul_tag_map tagmap;
protected:
  void virtual visit(SgNode *node);
public:
  void setTagMap(paul_tag_map ptm) { tagmap = ptm; }

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

void handle_comment(const string s, SgLocatedNode *node, paul_tag_map tagmap) {
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

    paul_tag_map::iterator ptm_it;

    ptm_it = tagmap.find(tag);
    
    if (ptm_it != tagmap.end()) {
      if ((*ptm_it).second == "key-value") {
        KVAnnotationValue *pValue = new KVAnnotationValue (value_text);

        // create the annotation
        Annotation *pAnn = new Annotation(value_text, node, tag, pValue);

        // add the annotation to the node:
        node->addNewAttribute (annotAttributeString, pAnn);

        // tracing for now:
        cerr << "KV --- Tag: " << pAnn->getTag()
             << " ; Value: " << pAnn->getValueString()
             << endl;
      } else if ((*ptm_it).second == "s-expression") {
        SXAnnotationValue *pValue = new SXAnnotationValue (value_text);

        // create the annotation
        Annotation *pAnn = new Annotation(value_text, node, tag, pValue);

        // add the annotation to the node:
        node->addNewAttribute (annotAttributeString, pAnn);

        // tracing for now:
        cerr << "SX --- Tag: " << pAnn->getTag()
             << " ; Value: " << pAnn->getValueString()
             << endl;

      } else {
        cerr << "UNSUPPORTED ANNOTATION FORMAT :: " << (*ptm_it).second << endl;
      }
    } else {
      // tag wasn't found
      cerr << "Tag (" << tag << ") encountered not present in configuration file." << endl;
    }


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
            handle_comment(comment,locatedNode,tagmap);
            break;
          }
          case CPP_COMMENT: {
            string comment = remove_cpp_comment_marks((*i)->getString());
            handle_comment(comment,locatedNode,tagmap);
            break;
          }
        }
      }
    }
  }
}

//////////////////////////

void paulDecorate (SgProject* project, string conf_fname)
{
  CommentVisitor v;

  if (conf_fname != "") {
    paul_tag_map ptm = read_paul_conf(conf_fname);
    v.setTagMap(ptm);
  }

  v.traverseInputFiles(project,preorder);    // FIXME:?
}
