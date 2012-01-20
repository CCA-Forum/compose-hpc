#ifndef __PAULCONFREADER_H__
#define __PAULCONFREADER_H__

#include <map>
#include <string>

using namespace std;

typedef map<const string, string> paul_tag_map;

paul_tag_map read_paul_conf(string fname);

#endif // __PAULCONFREADER_H__ 
