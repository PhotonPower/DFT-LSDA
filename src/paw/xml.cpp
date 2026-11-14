#include "paw/xml.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace lsda {

const XmlNode* XmlNode::child(const std::string& n) const {
  for (const auto& c : children)
    if (c->name == n) return c.get();
  return nullptr;
}

std::vector<const XmlNode*> XmlNode::children_named(const std::string& n) const {
  std::vector<const XmlNode*> out;
  for (const auto& c : children)
    if (c->name == n) out.push_back(c.get());
  return out;
}

std::string XmlNode::str(const std::string& key) const {
  auto it = attr.find(key);
  LSDA_CHECK(it != attr.end(), "XML: missing attribute '" + key + "' in <" + name + ">");
  return it->second;
}

double XmlNode::num(const std::string& key) const { return std::strtod(str(key).c_str(), nullptr); }
double XmlNode::num(const std::string& key, double fallback) const { return has(key) ? num(key) : fallback; }

std::vector<double> XmlNode::values() const {
  std::vector<double> v;
  const char* p = text.c_str();
  char* end = nullptr;
  for (;;) {
    const double x = std::strtod(p, &end);
    if (end == p) break;
    v.push_back(x);
    p = end;
  }
  return v;
}

namespace {

struct Parser {
  const std::string& s;
  size_t pos = 0;

  void skip_ws() {
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
  }
  bool starts(const char* t) const { return s.compare(pos, std::char_traits<char>::length(t), t) == 0; }
  void skip_until(const char* t) {
    const size_t e = s.find(t, pos);
    LSDA_CHECK(e != std::string::npos, "XML: unterminated construct");
    pos = e + std::char_traits<char>::length(t);
  }
  std::string read_name() {
    const size_t b = pos;
    while (pos < s.size() && (std::isalnum(static_cast<unsigned char>(s[pos])) || s[pos] == '_' || s[pos] == '-' ||
                              s[pos] == ':' || s[pos] == '.'))
      ++pos;
    return s.substr(b, pos - b);
  }
  std::string trim(const std::string& t) {
    size_t b = 0, e = t.size();
    while (b < e && std::isspace(static_cast<unsigned char>(t[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(t[e - 1]))) --e;
    return t.substr(b, e - b);
  }

  // parses one element; pos is at '<' of the start tag
  std::unique_ptr<XmlNode> element() {
    auto node = std::make_unique<XmlNode>();
    ++pos;  // '<'
    node->name = read_name();
    for (;;) {  // attributes
      skip_ws();
      LSDA_CHECK(pos < s.size(), "XML: unexpected end of file");
      if (starts("/>")) {
        pos += 2;
        return node;
      }
      if (s[pos] == '>') {
        ++pos;
        break;
      }
      const std::string key = read_name();
      LSDA_CHECK(!key.empty(), "XML: malformed attribute in <" + node->name + ">");
      skip_ws();
      LSDA_CHECK(s[pos] == '=', "XML: expected '=' after attribute " + key);
      ++pos;
      skip_ws();
      const char quote = s[pos];
      LSDA_CHECK(quote == '"' || quote == '\'', "XML: expected quoted attribute value");
      const size_t e = s.find(quote, pos + 1);
      LSDA_CHECK(e != std::string::npos, "XML: unterminated attribute value");
      node->attr[key] = trim(s.substr(pos + 1, e - pos - 1));
      pos = e + 1;
    }
    for (;;) {  // content
      const size_t lt = s.find('<', pos);
      LSDA_CHECK(lt != std::string::npos, "XML: missing end tag for <" + node->name + ">");
      node->text.append(s, pos, lt - pos);
      pos = lt;
      if (starts("<!--")) {
        skip_until("-->");
      } else if (starts("</")) {
        skip_until(">");
        return node;
      } else if (starts("<?")) {
        skip_until("?>");
      } else {
        node->children.push_back(element());
      }
    }
  }
};

}  // namespace

std::unique_ptr<XmlNode> parse_xml_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  LSDA_CHECK(in.good(), "cannot open file '" + path + "'");
  std::stringstream buf;
  buf << in.rdbuf();
  const std::string s = buf.str();
  Parser p{s};
  for (;;) {  // skip prolog and comments
    p.skip_ws();
    LSDA_CHECK(p.pos < s.size(), "XML: no root element in " + path);
    if (p.starts("<?"))
      p.skip_until("?>");
    else if (p.starts("<!--"))
      p.skip_until("-->");
    else if (p.starts("<!"))
      p.skip_until(">");
    else
      break;
  }
  return p.element();
}

}  // namespace lsda
