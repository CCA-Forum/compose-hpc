#include <iostream>
#include "rose.h"
#include "CommentVisitor.h"
#include "Annotation.h"
#include "Transforms.h"

#define C_COMMENT         (1)
#define CPP_COMMENT       (2)
#define ANNOTATION_PREFIX ('%')

using namespace std;

extern SgProject *root;

CommentVisitor::CommentVisitor(string inputFile,int * fcount) {
	inpFile = inputFile;
	fileCount = fcount;
}

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

void handle_comment(const string s, SgLocatedNode *node, string inpFile, int *fileCount) {
  if(is_annotation(s)) {
    string ann_text = annotation_text(s);
    Annotation *ann = Annotation::parse(ann_text);
    if(ann != NULL) {
      cerr << "Handling " << ann->get_id() << endl;
      Transform *transf = Transform::get_transform(node,ann);
      transf->generate(inpFile,fileCount);
      // AstDOTGeneration gen;
      // gen.generate(node,"boids.c");
      delete transf;
      delete ann;
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
            handle_comment(comment,locatedNode,inpFile,fileCount);
            break;
          }
          case CPP_COMMENT: {
            string comment = remove_cpp_comment_marks((*i)->getString());
            handle_comment(comment,locatedNode,inpFile,fileCount);
            break;
          }
        }
      }
    }
  }
}
