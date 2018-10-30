// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// FST definitions.

#include <fst/fst.h>

#include <sstream>

#include <fst/flags.h>
#include <fst/log.h>
#include <fst/matcher-fst.h>  // declarations of *_lookahead_fst_type

// FST flag definitions.

DEFINE_bool(fst_verify_properties, false,
            "Verify FST properties queried by TestProperties");

DEFINE_bool(fst_default_cache_gc, true, "Enable garbage collection of cache");

DEFINE_int64(fst_default_cache_gc_limit, 1 << 20LL,
             "Cache byte size that triggers garbage collection");

DEFINE_bool(fst_align, false, "Write FST data aligned where appropriate");

DEFINE_string(save_relabel_ipairs, "", "Save input relabel pairs to file");
DEFINE_string(save_relabel_opairs, "", "Save output relabel pairs to file");

DEFINE_string(fst_read_mode, "read",
              "Default file reading mode for mappable files");

namespace fst {

// FST type definitions for lookahead FSTs.
const char arc_lookahead_fst_type[] = "arc_lookahead";
const char ilabel_lookahead_fst_type[] = "ilabel_lookahead";
const char olabel_lookahead_fst_type[] = "olabel_lookahead";

// Identifies stream data as an FST (and its endianity).
constexpr int32 kFstMagicNumber = 2125659606;

// Checks for FST magic number in stream, to indicate caller function that the
// stream content is an FST header.
bool IsFstHeader(std::istream &strm, const string &source) {
  int64 pos = strm.tellg();
  bool match = true;
  int32 magic_number = 0;
  ReadType(strm, &magic_number);
  if (magic_number != kFstMagicNumber) {
      match = false;
  }
  strm.seekg(pos);
  return match;
}

// Checks FST magic number and reads in the header; if rewind = true,
// the stream is repositioned before call if possible.
bool FstHeader::Read(std::istream &strm, const string &source, bool rewind) {
  int64 pos = 0;
  if (rewind) pos = strm.tellg();
  int32 magic_number = 0;
  ReadType(strm, &magic_number);
  if (magic_number != kFstMagicNumber) {
      LOG(ERROR) << "FstHeader::Read: Bad FST header: " << source;
      if (rewind) strm.seekg(pos);
      return false;
  }
  ReadType(strm, &fsttype_);
  ReadType(strm, &arctype_);
  ReadType(strm, &version_);
  ReadType(strm, &flags_);
  ReadType(strm, &properties_);
  ReadType(strm, &start_);
  ReadType(strm, &numstates_);
  ReadType(strm, &numarcs_);
  if (!strm) {
    LOG(ERROR) << "FstHeader::Read: Read failed: " << source;
    return false;
  }
  if (rewind) strm.seekg(pos);
  return true;
}

// Writes FST magic number and FST header.
bool FstHeader::Write(std::ostream &strm, const string &source) const {
  WriteType(strm, kFstMagicNumber);
  WriteType(strm, fsttype_);
  WriteType(strm, arctype_);
  WriteType(strm, version_);
  WriteType(strm, flags_);
  WriteType(strm, properties_);
  WriteType(strm, start_);
  WriteType(strm, numstates_);
  WriteType(strm, numarcs_);
  return true;
}

string FstHeader::DebugString() const {
  std::ostringstream ostrm;
  ostrm << "fsttype: \"" << fsttype_ << "\" arctype: \"" << arctype_
        << "\" version: \"" << version_ << "\" flags: \"" << flags_
        << "\" properties: \"" << properties_ << "\" start: \"" << start_
        << "\" numstates: \"" << numstates_ << "\" numarcs: \"" << numarcs_
        << "\"";
  return ostrm.str();
}

FstReadOptions::FstReadOptions(const string &source, const FstHeader *header,
                               const SymbolTable *isymbols,
                               const SymbolTable *osymbols)
    : source(source),
      header(header),
      isymbols(isymbols),
      osymbols(osymbols),
      read_isymbols(true),
      read_osymbols(true) {
  mode = ReadMode(FLAGS_fst_read_mode);
}

FstReadOptions::FstReadOptions(const string &source,
                               const SymbolTable *isymbols,
                               const SymbolTable *osymbols)
    : source(source),
      header(nullptr),
      isymbols(isymbols),
      osymbols(osymbols),
      read_isymbols(true),
      read_osymbols(true) {
  mode = ReadMode(FLAGS_fst_read_mode);
}

FstReadOptions::FileReadMode FstReadOptions::ReadMode(const string &mode) {
  if (mode == "read") return READ;
  if (mode == "map") return MAP;
  LOG(ERROR) << "Unknown file read mode " << mode;
  return READ;
}

string FstReadOptions::DebugString() const {
  std::ostringstream ostrm;
  ostrm << "source: \"" << source << "\" mode: \""
        << (mode == READ ? "READ" : "MAP") << "\" read_isymbols: \""
        << (read_isymbols ? "true" : "false") << "\" read_osymbols: \""
        << (read_osymbols ? "true" : "false") << "\" header: \""
        << (header ? "set" : "null") << "\" isymbols: \""
        << (isymbols ? "set" : "null") << "\" osymbols: \""
        << (osymbols ? "set" : "null") << "\"";
  return ostrm.str();
}

}  // namespace fst
