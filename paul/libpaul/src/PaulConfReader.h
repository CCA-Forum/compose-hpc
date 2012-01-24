#ifndef __PAULCONFREADER_H__
#define __PAULCONFREADER_H__

#include <map>
#include <string>

/**
 * The paul_tag_map type simply defines a mapping between tag names
 * and the parser type.  In a PAUL structured comment, the format is:
 *
 *  %TAG [body]
 *
 * TAG can be any string that contains no whitespace.  All text following
 * the TAG to the end of the comment is treated as text that can be
 * consumed by the parser associated with comments of type TAG.
 */
typedef std::map<const std::string, std::string> paul_tag_map;

/**
 * This function reads the specified PAUL configuration file and
 * returns a paul_tag_map for use within the PaulDecorate code.
 *
 * PAUL config files are formatted as follows:  any line that begins with
 * a semi-colon is treated as a comment and ignored.  Every other line is
 * assumed to be formatted as:
 *
 *   TAG  parser-name
 *
 * Whitespace before the TAG is ignored.
 *
 * Valid parser-names include:
 *
 * - Plain
 * - s-expression
 * - key-value
 *
 * \param fname Filename of parameter file.
 * \return The paul_tag_map structure.
 */
paul_tag_map read_paul_conf(std::string fname);

#endif // __PAULCONFREADER_H__ 
