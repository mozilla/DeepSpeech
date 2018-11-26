#include "lm/binary_format.hh"

#include "lm/lm_exception.hh"
#include "util/file.hh"
#include "util/file_piece.hh"

#include <cstddef>
#include <cstring>
#include <limits>
#include <string>
#include <cstdlib>

#include <stdint.h>

namespace lm {
namespace ngram {

const char *kModelNames[6] = {"probing hash tables", "probing hash tables with rest costs", "trie", "trie with quantization", "trie with array-compressed pointers", "trie with quantization and array-compressed pointers"};

namespace {
const char kMagicBeforeVersion[] = "mmap lm http://kheafield.com/code format version";
const char kMagicBytes[] = "mmap lm http://kheafield.com/code format version 5\n\0";
// This must be shorter than kMagicBytes and indicates an incomplete binary file (i.e. build failed).
const char kMagicIncomplete[] = "mmap lm http://kheafield.com/code incomplete\n";
const long int kMagicVersion = 5;

// Old binary files built on 32-bit machines have this header.
// TODO: eliminate with next binary release.
struct OldSanity {
  char magic[sizeof(kMagicBytes)];
  float zero_f, one_f, minus_half_f;
  WordIndex one_word_index, max_word_index;
  uint64_t one_uint64;

  void SetToReference() {
    std::memset(this, 0, sizeof(OldSanity));
    std::memcpy(magic, kMagicBytes, sizeof(magic));
    zero_f = 0.0; one_f = 1.0; minus_half_f = -0.5;
    one_word_index = 1;
    max_word_index = std::numeric_limits<WordIndex>::max();
    one_uint64 = 1;
  }
};


// Test values aligned to 8 bytes.
struct Sanity {
  char magic[ALIGN8(sizeof(kMagicBytes))];
  float zero_f, one_f, minus_half_f;
  WordIndex one_word_index, max_word_index, padding_to_8;
  uint64_t one_uint64;

  void SetToReference() {
    std::memset(this, 0, sizeof(Sanity));
    std::memcpy(magic, kMagicBytes, sizeof(kMagicBytes));
    zero_f = 0.0; one_f = 1.0; minus_half_f = -0.5;
    one_word_index = 1;
    max_word_index = std::numeric_limits<WordIndex>::max();
    padding_to_8 = 0;
    one_uint64 = 1;
  }
};

std::size_t TotalHeaderSize(unsigned char order) {
  return ALIGN8(sizeof(Sanity) + sizeof(FixedWidthParameters) + sizeof(uint64_t) * order);
}

void WriteHeader(void *to, const Parameters &params) {
  Sanity header = Sanity();
  header.SetToReference();
  std::memcpy(to, &header, sizeof(Sanity));
  char *out = reinterpret_cast<char*>(to) + sizeof(Sanity);

  *reinterpret_cast<FixedWidthParameters*>(out) = params.fixed;
  out += sizeof(FixedWidthParameters);

  uint64_t *counts = reinterpret_cast<uint64_t*>(out);
  for (std::size_t i = 0; i < params.counts.size(); ++i) {
    counts[i] = params.counts[i];
  }
}

} // namespace

bool IsBinaryFormat(int fd) {
  const uint64_t size = util::SizeFile(fd);
  if (size == util::kBadSize || (size <= static_cast<uint64_t>(sizeof(Sanity)))) return false;
  // Try reading the header.
  util::scoped_memory memory;
  try {
    util::MapRead(util::LAZY, fd, 0, sizeof(Sanity), memory);
  } catch (const util::Exception &e) {
    return false;
  }
  Sanity reference_header = Sanity();
  reference_header.SetToReference();
  if (!std::memcmp(memory.get(), &reference_header, sizeof(Sanity))) return true;
  if (!std::memcmp(memory.get(), kMagicIncomplete, strlen(kMagicIncomplete))) {
    UTIL_THROW(FormatLoadException, "This binary file did not finish building");
  }
  if (!std::memcmp(memory.get(), kMagicBeforeVersion, strlen(kMagicBeforeVersion))) {
    char *end_ptr;
    const char *begin_version = static_cast<const char*>(memory.get()) + strlen(kMagicBeforeVersion);
    long int version = std::strtol(begin_version, &end_ptr, 10);
    if ((end_ptr != begin_version) && version != kMagicVersion) {
      UTIL_THROW(FormatLoadException, "Binary file has version " << version << " but this implementation expects version " << kMagicVersion << " so you'll have to use the ARPA to rebuild your binary");
    }

    OldSanity old_sanity = OldSanity();
    old_sanity.SetToReference();
    UTIL_THROW_IF(!std::memcmp(memory.get(), &old_sanity, sizeof(OldSanity)), FormatLoadException, "Looks like this is an old 32-bit format.  The old 32-bit format has been removed so that 64-bit and 32-bit files are exchangeable.");
    UTIL_THROW(FormatLoadException, "File looks like it should be loaded with mmap, but the test values don't match.  Try rebuilding the binary format LM using the same code revision, compiler, and architecture");
  }
  return false;
}

void ReadHeader(int fd, Parameters &out) {
  util::SeekOrThrow(fd, sizeof(Sanity));
  util::ReadOrThrow(fd, &out.fixed, sizeof(out.fixed));
  if (out.fixed.probing_multiplier < 1.0)
    UTIL_THROW(FormatLoadException, "Binary format claims to have a probing multiplier of " << out.fixed.probing_multiplier << " which is < 1.0.");

  out.counts.resize(static_cast<std::size_t>(out.fixed.order));
  if (out.fixed.order) util::ReadOrThrow(fd, &*out.counts.begin(), sizeof(uint64_t) * out.fixed.order);
}

void MatchCheck(ModelType model_type, unsigned int search_version, const Parameters &params) {
  if (params.fixed.model_type != model_type) {
    if (static_cast<unsigned int>(params.fixed.model_type) >= (sizeof(kModelNames) / sizeof(const char *)))
      UTIL_THROW(FormatLoadException, "The binary file claims to be model type " << static_cast<unsigned int>(params.fixed.model_type) << " but this is not implemented for in this inference code.");
    UTIL_THROW(FormatLoadException, "The binary file was built for " << kModelNames[params.fixed.model_type] << " but the inference code is trying to load " << kModelNames[model_type]);
  }
  UTIL_THROW_IF(search_version != params.fixed.search_version, FormatLoadException, "The binary file has " << kModelNames[params.fixed.model_type] << " version " << params.fixed.search_version << " but this code expects " << kModelNames[params.fixed.model_type] << " version " << search_version);
}

const std::size_t kInvalidSize = static_cast<std::size_t>(-1);

BinaryFormat::BinaryFormat(const Config &config)
  : write_method_(config.write_method), write_mmap_(config.write_mmap), load_method_(config.load_method),
    header_size_(kInvalidSize), vocab_size_(kInvalidSize), vocab_string_offset_(kInvalidOffset) {}

void BinaryFormat::InitializeBinary(int fd, ModelType model_type, unsigned int search_version, Parameters &params) {
  file_.reset(fd);
  write_mmap_ = NULL; // Ignore write requests; this is already in binary format.
  ReadHeader(fd, params);
  MatchCheck(model_type, search_version, params);
  header_size_ = TotalHeaderSize(params.counts.size());
}

void BinaryFormat::ReadForConfig(void *to, std::size_t amount, uint64_t offset_excluding_header) const {
  assert(header_size_ != kInvalidSize);
  util::ErsatzPRead(file_.get(), to, amount, offset_excluding_header + header_size_);
}

void *BinaryFormat::LoadBinary(std::size_t size) {
  assert(header_size_ != kInvalidSize);
  const uint64_t file_size = util::SizeFile(file_.get());
  // The header is smaller than a page, so we have to map the whole header as well.
  uint64_t total_map = static_cast<uint64_t>(header_size_) + static_cast<uint64_t>(size);
  UTIL_THROW_IF(file_size != util::kBadSize && file_size < total_map, FormatLoadException, "Binary file has size " << file_size << " but the headers say it should be at least " << total_map);

  util::MapRead(load_method_, file_.get(), 0, util::CheckOverflow(total_map), mapping_);

  vocab_string_offset_ = total_map;
  return reinterpret_cast<uint8_t*>(mapping_.get()) + header_size_;
}

void *BinaryFormat::SetupJustVocab(std::size_t memory_size, uint8_t order) {
  vocab_size_ = memory_size;
  if (!write_mmap_) {
    header_size_ = 0;
    util::HugeMalloc(memory_size, true, memory_vocab_);
    return reinterpret_cast<uint8_t*>(memory_vocab_.get());
  }
  header_size_ = TotalHeaderSize(order);
  std::size_t total = util::CheckOverflow(static_cast<uint64_t>(header_size_) + static_cast<uint64_t>(memory_size));
  file_.reset(util::CreateOrThrow(write_mmap_));
  // some gccs complain about uninitialized variables even though all enum values are covered.
  void *vocab_base = NULL;
  switch (write_method_) {
    case Config::WRITE_MMAP:
      mapping_.reset(util::MapZeroedWrite(file_.get(), total), total, util::scoped_memory::MMAP_ALLOCATED);
      util::AdviseHugePages(vocab_base, total);
      vocab_base = mapping_.get();
      break;
    case Config::WRITE_AFTER:
      util::ResizeOrThrow(file_.get(), 0);
      util::HugeMalloc(total, true, memory_vocab_);
      vocab_base = memory_vocab_.get();
      break;
  }
  strncpy(reinterpret_cast<char*>(vocab_base), kMagicIncomplete, header_size_);
  return reinterpret_cast<uint8_t*>(vocab_base) + header_size_;
}

void *BinaryFormat::GrowForSearch(std::size_t memory_size, std::size_t vocab_pad, void *&vocab_base) {
  assert(vocab_size_ != kInvalidSize);
  vocab_pad_ = vocab_pad;
  std::size_t new_size = header_size_ + vocab_size_ + vocab_pad_ + memory_size;
  vocab_string_offset_ = new_size;
  if (!write_mmap_ || write_method_ == Config::WRITE_AFTER) {
    util::HugeMalloc(memory_size, true, memory_search_);
    assert(header_size_ == 0 || write_mmap_);
    vocab_base = reinterpret_cast<uint8_t*>(memory_vocab_.get()) + header_size_;
    util::AdviseHugePages(memory_search_.get(), memory_size);
    return reinterpret_cast<uint8_t*>(memory_search_.get());
  }

  assert(write_method_ == Config::WRITE_MMAP);
  // Also known as total size without vocab words.
  // Grow the file to accomodate the search, using zeros.
  // According to man mmap, behavior is undefined when the file is resized
  // underneath a mmap that is not a multiple of the page size.  So to be
  // safe, we'll unmap it and map it again.
  mapping_.reset();
  util::ResizeOrThrow(file_.get(), new_size);
  void *ret;
  MapFile(vocab_base, ret);
  util::AdviseHugePages(ret, new_size);
  return ret;
}

void BinaryFormat::WriteVocabWords(const std::string &buffer, void *&vocab_base, void *&search_base) {
  // Checking Config's include_vocab is the responsibility of the caller.
  assert(header_size_ != kInvalidSize && vocab_size_ != kInvalidSize);
  if (!write_mmap_) {
    // Unchanged base.
    vocab_base = reinterpret_cast<uint8_t*>(memory_vocab_.get());
    search_base = reinterpret_cast<uint8_t*>(memory_search_.get());
    return;
  }
  if (write_method_ == Config::WRITE_MMAP) {
    mapping_.reset();
  }
  util::SeekOrThrow(file_.get(), VocabStringReadingOffset());
  util::WriteOrThrow(file_.get(), &buffer[0], buffer.size());
  if (write_method_ == Config::WRITE_MMAP) {
    MapFile(vocab_base, search_base);
  } else {
    vocab_base = reinterpret_cast<uint8_t*>(memory_vocab_.get()) + header_size_;
    search_base = reinterpret_cast<uint8_t*>(memory_search_.get());
  }
}

void BinaryFormat::FinishFile(const Config &config, ModelType model_type, unsigned int search_version, const std::vector<uint64_t> &counts) {
  if (!write_mmap_) return;
  switch (write_method_) {
    case Config::WRITE_MMAP:
      util::SyncOrThrow(mapping_.get(), mapping_.size());
      break;
    case Config::WRITE_AFTER:
      util::SeekOrThrow(file_.get(), 0);
      util::WriteOrThrow(file_.get(), memory_vocab_.get(), memory_vocab_.size());
      util::SeekOrThrow(file_.get(), header_size_ + vocab_size_ + vocab_pad_);
      util::WriteOrThrow(file_.get(), memory_search_.get(), memory_search_.size());
      util::FSyncOrThrow(file_.get());
      break;
  }
  // header and vocab share the same mmap.
  Parameters params = Parameters();
  memset(&params, 0, sizeof(Parameters));
  params.counts = counts;
  params.fixed.order = counts.size();
  params.fixed.probing_multiplier = config.probing_multiplier;
  params.fixed.model_type = model_type;
  params.fixed.has_vocabulary = config.include_vocab;
  params.fixed.search_version = search_version;
  switch (write_method_) {
    case Config::WRITE_MMAP:
      WriteHeader(mapping_.get(), params);
      util::SyncOrThrow(mapping_.get(), mapping_.size());
      break;
    case Config::WRITE_AFTER:
      {
        std::vector<uint8_t> buffer(TotalHeaderSize(counts.size()));
        WriteHeader(&buffer[0], params);
        util::SeekOrThrow(file_.get(), 0);
        util::WriteOrThrow(file_.get(), &buffer[0], buffer.size());
      }
      break;
  }
}

void BinaryFormat::MapFile(void *&vocab_base, void *&search_base) {
  mapping_.reset(util::MapOrThrow(vocab_string_offset_, true, util::kFileFlags, false, file_.get()), vocab_string_offset_, util::scoped_memory::MMAP_ALLOCATED);
  vocab_base = reinterpret_cast<uint8_t*>(mapping_.get()) + header_size_;
  search_base = reinterpret_cast<uint8_t*>(mapping_.get()) + header_size_ + vocab_size_ + vocab_pad_;
}

bool RecognizeBinary(const char *file, ModelType &recognized) {
  util::scoped_fd fd(util::OpenReadOrThrow(file));
  if (!IsBinaryFormat(fd.get())) {
    return false;
  }
  Parameters params;
  ReadHeader(fd.get(), params);
  recognized = params.fixed.model_type;
  return true;
}

} // namespace ngram
} // namespace lm
