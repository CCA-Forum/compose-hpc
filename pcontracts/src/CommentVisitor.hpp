/*
 * File:         CommentVisitor.hpp
 * Description:  TEMPORARY variant of PAUL's CommentVisitor in PaulDecorate.cpp,
 *               at least until there is one -- preferably with a map allowing
 *               the addition of parser names to associated parser function(s).
 * Source:       Portion of contents of PaulDecorate.cpp
 */
#ifndef included_CommentVisitor_hpp
#define included_CommentVisitor_hpp

#include "rose.h"
#include "PaulConfReader.h"

/**
 * Class:  CommentVisitor
 */
class CommentVisitor : public AstSimpleProcessing
{
  private:
    paul_tag_map tagmap;
  
  protected:
    void virtual visit(SgNode *node);

    void handle_comment(const string s, SgLocatedNode *node, 
                        paul_tag_map tagmap);
  
  public:
    /* Default constructor */
    CommentVisitor() {}
};

#endif /* included_CommentVisitor_hpp */
