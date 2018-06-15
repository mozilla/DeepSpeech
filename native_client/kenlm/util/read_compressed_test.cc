#include "util/read_compressed.hh"

#include "util/file.hh"
#include "util/have.hh"

#define BOOST_TEST_MODULE ReadCompressedTest
#include <boost/test/unit_test.hpp>
#include <boost/scoped_ptr.hpp>

#include <fstream>
#include <string>
#include <cstdlib>

#if defined __MINGW32__
#include <ctime>
#include <fcntl.h>

#if !defined mkstemp
// TODO insecure
int mkstemp(char * stemplate)
{
    char *filename = mktemp(stemplate);
    if (filename == NULL)
        return -1;
    return open(filename, O_RDWR | O_CREAT, 0600);
}
#endif

#endif // defined

namespace util {
namespace {

void ReadLoop(ReadCompressed &reader, void *to_void, std::size_t amount) {
  uint8_t *to = static_cast<uint8_t*>(to_void);
  while (amount) {
    std::size_t ret = reader.Read(to, amount);
    BOOST_REQUIRE(ret);
    to += ret;
    amount -= ret;
  }
}

const uint32_t kSize4 = 100000 / 4;

std::string WriteRandom() {
  char name[] = "tempXXXXXX";
  scoped_fd original(mkstemp(name));
  BOOST_REQUIRE(original.get() > 0);
  for (uint32_t i = 0; i < kSize4; ++i) {
    WriteOrThrow(original.get(), &i, sizeof(uint32_t));
  }
  return name;
}

void VerifyRead(ReadCompressed &reader) {
  for (uint32_t i = 0; i < kSize4; ++i) {
    uint32_t got;
    ReadLoop(reader, &got, sizeof(uint32_t));
    BOOST_CHECK_EQUAL(i, got);
  }

  char ignored;
  BOOST_CHECK_EQUAL((std::size_t)0, reader.Read(&ignored, 1));
  // Test double EOF call.
  BOOST_CHECK_EQUAL((std::size_t)0, reader.Read(&ignored, 1));
}

void TestRandom(const char *compressor) {
  std::string name(WriteRandom());

  char gzname[] = "tempXXXXXX";
  scoped_fd gzipped(mkstemp(gzname));

  std::string command(compressor);
#ifdef __CYGWIN__
  command += ".exe";
#endif
  command += " <\"";
  command += name;
  command += "\" >\"";
  command += gzname;
  command += "\"";
  BOOST_REQUIRE_EQUAL(0, system(command.c_str()));

  BOOST_CHECK_EQUAL(0, unlink(name.c_str()));
  BOOST_CHECK_EQUAL(0, unlink(gzname));

  ReadCompressed reader(gzipped.release());
  VerifyRead(reader);
}

BOOST_AUTO_TEST_CASE(Uncompressed) {
  TestRandom("cat");
}

#ifdef HAVE_ZLIB
BOOST_AUTO_TEST_CASE(ReadGZ) {
  TestRandom("gzip");
}
#endif // HAVE_ZLIB

#ifdef HAVE_BZLIB
BOOST_AUTO_TEST_CASE(ReadBZ) {
  TestRandom("bzip2");
}
#endif // HAVE_BZLIB

#ifdef HAVE_XZLIB
BOOST_AUTO_TEST_CASE(ReadXZ) {
  TestRandom("xz");
}
#endif

#ifdef HAVE_ZLIB
BOOST_AUTO_TEST_CASE(AppendGZ) {
}
#endif

BOOST_AUTO_TEST_CASE(IStream) {
  std::string name(WriteRandom());
  std::fstream stream(name.c_str(), std::ios::in);
  BOOST_CHECK_EQUAL(0, unlink(name.c_str()));
  ReadCompressed reader;
  reader.Reset(stream);
  VerifyRead(reader);
}

} // namespace
} // namespace util
