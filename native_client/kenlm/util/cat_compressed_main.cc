// Like cat but interprets compressed files.
#include "util/file.hh"
#include "util/read_compressed.hh"

#include <cstring>
#include <iostream>

namespace {
const std::size_t kBufSize = 16384;
void Copy(util::ReadCompressed &from, int to) {
  util::scoped_malloc buffer(util::MallocOrThrow(kBufSize));
  while (std::size_t amount = from.Read(buffer.get(), kBufSize)) {
    util::WriteOrThrow(to, buffer.get(), amount);
  }
}
} // namespace

int main(int argc, char *argv[]) {
  // Lane Schwartz likes -h and --help
  for (int i = 1; i < argc; ++i) {
    char *arg = argv[i];
    if (!strcmp(arg, "--")) break;
    if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
      std::cerr <<
        "A cat implementation that interprets compressed files.\n"
        "Usage: " << argv[0] << " [file1] [file2] ...\n"
        "If no file is provided, then stdin is read.\n";
      return 1;
    }
  }

  try {
    if (argc == 1) {
      util::ReadCompressed in(0);
      Copy(in, 1);
    } else {
      for (int i = 1; i < argc; ++i) {
        util::ReadCompressed in(util::OpenReadOrThrow(argv[i]));
        Copy(in, 1);
      }
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 2;
  }
  return 0;
}
