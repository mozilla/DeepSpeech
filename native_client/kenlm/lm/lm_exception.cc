#include "lm/lm_exception.hh"

#include <cerrno>
#include <cstdio>

namespace lm {

ConfigException::ConfigException() throw() {}
ConfigException::~ConfigException() throw() {}

LoadException::LoadException() throw() {}
LoadException::~LoadException() throw() {}

FormatLoadException::FormatLoadException() throw() {}
FormatLoadException::~FormatLoadException() throw() {}

VocabLoadException::VocabLoadException() throw() {}
VocabLoadException::~VocabLoadException() throw() {}

SpecialWordMissingException::SpecialWordMissingException() throw() {}
SpecialWordMissingException::~SpecialWordMissingException() throw() {}

} // namespace lm
