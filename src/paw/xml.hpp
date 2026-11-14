// Minimal XML reader, sufficient for PAW-XML datasets (elements, attributes, text; no entities/CDATA).
#pragma once
#include <map>
#include <memory>

#include "base/common.hpp"

namespace lsda {

struct XmlNode {
  std::string name;
  std::map<std::string, std::string> attr;
  std::string text;
  std::vector<std::unique_ptr<XmlNode>> children;

  const XmlNode* child(const std::string& n) const;                 // first child or nullptr
  std::vector<const XmlNode*> children_named(const std::string& n) const;
  bool has(const std::string& key) const { return attr.count(key) > 0; }
  std::string str(const std::string& key) const;                    // required attribute
  double num(const std::string& key) const;                         // required numeric attribute
  double num(const std::string& key, double fallback) const;
  std::vector<double> values() const;                               // text parsed as numbers
};

std::unique_ptr<XmlNode> parse_xml_file(const std::string& path);

}  // namespace lsda
