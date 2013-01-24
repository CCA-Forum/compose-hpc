#include <iostream>
#include "rose.h"
#include "SXAnnotationValue.h"
#include "KVAnnotationValue.h"
#include "PlainAnnotationValue.h"
#include "PaulDecorate.h"
#include "PaulConfReader.h"
#include "Utilities.h"

#include <sys/wait.h>
#include <unistd.h>
#include <cstring>
#include <sstream>

#define C_COMMENT         (1)
#define CPP_COMMENT       (2)
#define FTN_COMMENT       (3)
#define F90_COMMENT       (4)
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
  return remove_leading_whitespace(s.substr(2));
}

string remove_c_comment_marks(const string s) {
  return remove_leading_whitespace(s.substr(2,s.size() - 4));
}

string remove_f_comment_marks(const string s) {
  return remove_leading_whitespace(s.substr(1));
}

bool is_annotation(const string s) {
  return s[0] == ANNOTATION_PREFIX;
}

string annotation_text(const string s) {
  return s.substr(1);
}

/// execute the command \c filter, write input to its stdin and return
/// the output of the command
/// TODO: execute each command only once and leave the pipes open
string exec_cmd(const string& filter, const string& input) {
  stringstream output;
  int d_read = 0;
  int d_write = 1;
  int p_in[2];
  int p_out[2];
  pipe(p_in);
  pipe(p_out);

  pid_t pid = fork();
  if (pid == 0) {
    // child process
    // point the child process' stdio the pipes
    dup2(p_in[d_read], 0); // stdin = 0
    dup2(p_out[d_write], 1); // stdout = 1
    close(p_in[d_write]);
    close(p_out[d_read]);
    char *arg0 = strdup(filter.c_str());
    char *argv[] = { arg0, NULL };
    execv(arg0, argv);
    cerr << "Could not execute \"" << filter << "\"\n";
    _exit(EXIT_FAILURE);
  } else {
    // parent process
    assert(pid > 0);
    close(p_in[d_read]);
    close(p_out[d_write]);

    // send the input to the filter command
    ssize_t size = input.size();
    while (size > 0) {
      ssize_t len = write(p_in[d_write], input.c_str(), size);
      if (len < 0)  {
	cerr<<"**WARNING: Could not write to child process"<<endl;
	break;
      }
      size -= len;
    }
    assert(size==0);
    close(p_in[d_write]);

    // grab the output of the command
    ssize_t len;
    do {
      char buf[1024];
      len = read(p_out[d_read], buf, 1024);
      if (len < 0) {
	cerr<<"**WARNING: Could not read from child process"<<endl;
	break;
      }
      //cerr<< "read "<<len<<" bytes"<<endl;
      output << string(buf, len);
    } while (len > 0);
    close(p_out[d_read]);

    // wait for the child process to terminate
    int status;
    if (waitpid(pid, &status, 0) == -1) {
      perror("waitpid");
      exit(EXIT_FAILURE);
    }
    if (WEXITSTATUS(status) != 0) {
      cerr<<"**WARNING: Command "<<filter
	  <<" returned with exit status "<<WEXITSTATUS(status)<<endl;
    }

  }
  cerr <<"read annotation "<<output.str()<<endl;
  return output.str();
}

void handle_comment(const string s, SgLocatedNode *node, paul_tag_map tagmap) {
  if(is_annotation(s)) {
    string ann_text = annotation_text(s);

    // split into the TAG and Value
    if (!ann_text.empty()) {
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

      // first, lookup annotation
      Annotation *pAnn = (Annotation *)node->getAttribute(tag);
    
      if (ptm_it != tagmap.end()) {
        if ((*ptm_it).second == "key-value") {
	  //
	  // key-value pair annotation
	  //
          KVAnnotationValue *pValue = new KVAnnotationValue (value_text);

	  if (pAnn == NULL) {
	    // create the annotation
	    pAnn = new Annotation(value_text, node, tag, pValue);

	    // add the annotation to the node:
	    node->addNewAttribute (tag, pAnn);
	  } else {
	    // need to merge with original annotation
	    KVAnnotationValue *original = (KVAnnotationValue *)pAnn->getValue();

	    // do the merge
	    original->merge(pValue);
	  }

        } else if ((*ptm_it).second == "s-expression") {
	  //
	  // s-expression annotation
	  //
          SXAnnotationValue *pValue = new SXAnnotationValue (value_text);

	  if (pAnn == NULL) {
	    // create the annotation
	    pAnn = new Annotation(value_text, node, tag, pValue);

	    // add the annotation to the node:
	    node->addNewAttribute (tag, pAnn);
	  } else {
	    // need to merge with original annotation
	    SXAnnotationValue *original = (SXAnnotationValue *)pAnn->getValue();

	    // do the merge
	    original->merge(pValue);
	  }

        } else if ((*ptm_it).second == "contract-clause") {
	  //
	  // [interface] contract annotation
	  //
	  // 1) Instantiate Annotation value
	  // 2) Instantiate Annotation
	  // 3) Add the annotation to the node or merge with existing annotation
	  //
          cerr << "Contract clause parser annotations not yet supported\n";

        } else if ((*ptm_it).second == "plain") {
	  //
	  // plain annotation
	  //
	  PlainAnnotationValue *pValue = new PlainAnnotationValue(value_text);

	  if (pAnn == NULL) {
	    // create the annotation
	    pAnn = new Annotation(value_text, node, tag, pValue);

	    // add the annotation to the node:
	    node->addNewAttribute (tag, pAnn);
	  } else {
	    // need to merge with original annotation
	    PlainAnnotationValue *original = 
	      (PlainAnnotationValue *)pAnn->getValue();

	    // do the merge
	    original->merge(pValue);
	  }

	} else if ((*ptm_it).second[0] == '|') {
	  //
	  // PIPE'ed annotation: after the '|' comes the name of a
	  // filter (program) that translates the annotation into
	  // s-expressions.
	  //
	  string filter = (*ptm_it).second.substr(1);
          cerr << "Establishing pipe to "<< filter <<"\n";

	  string sexpr = exec_cmd(filter, value_text);

	  SXAnnotationValue *pValue = new SXAnnotationValue (sexpr);

	  if (pAnn == NULL) {
	    // create the annotation
	    pAnn = new Annotation(value_text, node, tag, pValue);

	    // add the annotation to the node:
	    node->addNewAttribute (tag, pAnn);
	  } else {
	    // need to merge with original annotation
	    SXAnnotationValue *original = (SXAnnotationValue *)pAnn->getValue();

	    // do the merge
	    original->merge(pValue);
	  }
	} else {
	  cerr << "UNSUPPORTED ANNOTATION FORMAT (NON-FATAL, IGNORING) :: Tag="
	       << tag << ", Parser=" << (*ptm_it).second << endl;
	}
      } else {
	// tag wasn't found
	cerr << "Tag (" << tag
	     << ") encountered not present in configuration file." << endl;
      }
    } else {
      // empty annotation
      cerr << "Empty annotation encountered." << endl;
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
          case FTN_COMMENT:
          case F90_COMMENT: {
            string comment = remove_f_comment_marks((*i)->getString());
            handle_comment(comment,locatedNode,tagmap);
            break;
	  }
	  default:
	    // ignore other non-comment preprocessor stuff
	    break;
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
