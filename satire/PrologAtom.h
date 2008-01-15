/* -*- C++ -*-
Copyright 2006 Christoph Bonitz (christoph.bonitz@gmail.com)
see LICENSE in the root folder of this project
*/
#ifndef PROLOGATOM_H_
#define PROLOGATOM_H_
#include "PrologTerm.h"
#include <iostream>
#include <ctype.h>

/**class representing a prolog atom*/
class PrologAtom : public PrologTerm {
public:
  ///the destructor
  ~PrologAtom() {};	
  ///constructor setting the string
  PrologAtom(std::string name) {
#if 0
    // FIXME: qualified names should be deactivated anyway
    // toProlog (ok) vs. main.C (err)
    if ((name.length() > 2) &&
	(name[0] == name[1] == ':'))
      mName = name.substr(2, name.length());
    else 
#endif
      mName = name;
  };
  ///the arity is always 0
  int getArity() {return 0;};
  ///an atom is always ground
  bool isGround() {return true;};
  ///return the string
  std::string getName() {return mName;};
  /// return the string
  std::string getRepresentation() {
    // Quote Atom
    if ((mName.length() > 0) && isupper(mName[0])) {
      std::string s;
      s = "'" + mName  + "'";
      return s;
    }
    return mName;
  };
private:
  /// the string
  std::string mName;
};
#endif
