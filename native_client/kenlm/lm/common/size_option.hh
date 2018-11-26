#include <boost/program_options.hpp>

#include <cstddef>
#include <string>

namespace lm {

// Create a boost program option for data sizes.  This parses sizes like 1T and 10k.
boost::program_options::typed_value<std::string> *SizeOption(std::size_t &to, const char *default_value);

} // namespace lm
