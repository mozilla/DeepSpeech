// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// FST abstract base class definition, state and arc iterator interface, and
// suggested base implementation.

#ifndef FST_FST_H_
#define FST_FST_H_

#include <sys/types.h>

#include <cmath>
#include <cstddef>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include <fst/compat.h>
#include <fst/flags.h>
#include <fst/log.h>
#include <fstream>

#include <fst/arc.h>
#include <fst/memory.h>
#include <fst/properties.h>
#include <fst/register.h>
#include <fst/symbol-table.h>
#include <fst/util.h>


DECLARE_bool(fst_align);

namespace fst {

bool IsFstHeader(std::istream &, const string &);

class FstHeader;

template <class Arc>
struct StateIteratorData;

template <class Arc>
struct ArcIteratorData;

template <class Arc>
class MatcherBase;

struct FstReadOptions {
  // FileReadMode(s) are advisory, there are many conditions than prevent a
  // file from being mapped, READ mode will be selected in these cases with
  // a warning indicating why it was chosen.
  enum FileReadMode { READ, MAP };

  string source;                // Where you're reading from.
  const FstHeader *header;      // Pointer to FST header; if non-zero, use
                                // this info (don't read a stream header).
  const SymbolTable *isymbols;  // Pointer to input symbols; if non-zero, use
                                // this info (read and skip stream isymbols)
  const SymbolTable *osymbols;  // Pointer to output symbols; if non-zero, use
                                // this info (read and skip stream osymbols)
  FileReadMode mode;            // Read or map files (advisory, if possible)
  bool read_isymbols;           // Read isymbols, if any (default: true).
  bool read_osymbols;           // Read osymbols, if any (default: true).

  explicit FstReadOptions(const string &source = "<unspecified>",
                          const FstHeader *header = nullptr,
                          const SymbolTable *isymbols = nullptr,
                          const SymbolTable *osymbols = nullptr);

  explicit FstReadOptions(const string &source, const SymbolTable *isymbols,
                          const SymbolTable *osymbols = nullptr);

  // Helper function to convert strings FileReadModes into their enum value.
  static FileReadMode ReadMode(const string &mode);

  // Outputs a debug string for the FstReadOptions object.
  string DebugString() const;
};

struct FstWriteOptions {
  string source;        // Where you're writing to.
  bool write_header;    // Write the header?
  bool write_isymbols;  // Write input symbols?
  bool write_osymbols;  // Write output symbols?
  bool align;           // Write data aligned (may fail on pipes)?
  bool stream_write;    // Avoid seek operations in writing.

  explicit FstWriteOptions(const string &source = "<unspecifed>",
                           bool write_header = true, bool write_isymbols = true,
                           bool write_osymbols = true,
                           bool align = FLAGS_fst_align,
                           bool stream_write = false)
      : source(source),
        write_header(write_header),
        write_isymbols(write_isymbols),
        write_osymbols(write_osymbols),
        align(align),
        stream_write(stream_write) {}
};

// Header class.
//
// This is the recommended file header representation.

class FstHeader {
 public:
  enum {
    HAS_ISYMBOLS = 0x1,  // Has input symbol table.
    HAS_OSYMBOLS = 0x2,  // Has output symbol table.
    IS_ALIGNED = 0x4,    // Memory-aligned (where appropriate).
  } Flags;

  FstHeader() : version_(0), flags_(0), properties_(0), start_(-1),
      numstates_(0), numarcs_(0) {}

  const string &FstType() const { return fsttype_; }

  const string &ArcType() const { return arctype_; }

  int32 Version() const { return version_; }

  int32 GetFlags() const { return flags_; }

  uint64 Properties() const { return properties_; }

  int64 Start() const { return start_; }

  int64 NumStates() const { return numstates_; }

  int64 NumArcs() const { return numarcs_; }

  void SetFstType(const string &type) { fsttype_ = type; }

  void SetArcType(const string &type) { arctype_ = type; }

  void SetVersion(int32 version) { version_ = version; }

  void SetFlags(int32 flags) { flags_ = flags; }

  void SetProperties(uint64 properties) { properties_ = properties; }

  void SetStart(int64 start) { start_ = start; }

  void SetNumStates(int64 numstates) { numstates_ = numstates; }

  void SetNumArcs(int64 numarcs) { numarcs_ = numarcs; }

  bool Read(std::istream &strm, const string &source,
            bool rewind = false);

  bool Write(std::ostream &strm, const string &source) const;

  // Outputs a debug string for the FstHeader object.
  string DebugString() const;

 private:
  string fsttype_;     // E.g. "vector".
  string arctype_;     // E.g. "standard".
  int32 version_;      // Type version number.
  int32 flags_;        // File format bits.
  uint64 properties_;  // FST property bits.
  int64 start_;        // Start state.
  int64 numstates_;    // # of states.
  int64 numarcs_;      // # of arcs.
};

// Specifies matcher action.
enum MatchType {
  MATCH_INPUT = 1,   // Match input label.
  MATCH_OUTPUT = 2,  // Match output label.
  MATCH_BOTH = 3,    // Match input or output label.
  MATCH_NONE = 4,    // Match nothing.
  MATCH_UNKNOWN = 5
};  // Otherwise, match type unknown.

constexpr int kNoLabel = -1;    // Not a valid label.
constexpr int kNoStateId = -1;  // Not a valid state ID.

// A generic FST, templated on the arc definition, with common-demoninator
// methods (use StateIterator and ArcIterator to iterate over its states and
// arcs).
template <class A>
class Fst {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  virtual ~Fst() {}

  // Initial state.
  virtual StateId Start() const = 0;

  // State's final weight.
  virtual Weight Final(StateId) const = 0;

  // State's arc count.
  virtual size_t NumArcs(StateId) const = 0;

  // State's input epsilon count.
  virtual size_t NumInputEpsilons(StateId) const = 0;

  // State's output epsilon count.
  virtual size_t NumOutputEpsilons(StateId) const = 0;

  // Property bits. If test = false, return stored properties bits for mask
  // (some possibly unknown); if test = true, return property bits for mask
  // (computing o.w. unknown).
  virtual uint64 Properties(uint64 mask, bool test) const = 0;

  // FST type name.
  virtual const string &Type() const = 0;

  // Gets a copy of this Fst. The copying behaves as follows:
  //
  // (1) The copying is constant time if safe = false or if safe = true
  // and is on an otherwise unaccessed FST.
  //
  // (2) If safe = true, the copy is thread-safe in that the original
  // and copy can be safely accessed (but not necessarily mutated) by
  // separate threads. For some FST types, 'Copy(true)' should only be
  // called on an FST that has not otherwise been accessed. Behavior is
  // otherwise undefined.
  //
  // (3) If a MutableFst is copied and then mutated, then the original is
  // unmodified and vice versa (often by a copy-on-write on the initial
  // mutation, which may not be constant time).
  virtual Fst<Arc> *Copy(bool safe = false) const = 0;

  // Reads an FST from an input stream; returns nullptr on error.
  static Fst<Arc> *Read(std::istream &strm, const FstReadOptions &opts) {
    FstReadOptions ropts(opts);
    FstHeader hdr;
    if (ropts.header) {
      hdr = *opts.header;
    } else {
      if (!hdr.Read(strm, opts.source)) return nullptr;
      ropts.header = &hdr;
    }
    const auto &fst_type = hdr.FstType();
    const auto reader = FstRegister<Arc>::GetRegister()->GetReader(fst_type);
    if (!reader) {
      LOG(ERROR) << "Fst::Read: Unknown FST type " << fst_type
                 << " (arc type = " << Arc::Type() << "): " << ropts.source;
      return nullptr;
    }
    return reader(strm, ropts);
  }

  // Reads an FST from a file; returns nullptr on error. An empty filename
  // results in reading from standard input.
  static Fst<Arc> *Read(const string &filename) {
    if (!filename.empty()) {
      std::ifstream strm(filename,
                              std::ios_base::in | std::ios_base::binary);
      if (!strm) {
        LOG(ERROR) << "Fst::Read: Can't open file: " << filename;
        return nullptr;
      }
      return Read(strm, FstReadOptions(filename));
    } else {
      return Read(std::cin, FstReadOptions("standard input"));
    }
  }

  // Writes an FST to an output stream; returns false on error.
  virtual bool Write(std::ostream &strm, const FstWriteOptions &opts) const {
    LOG(ERROR) << "Fst::Write: No write stream method for " << Type()
               << " FST type";
    return false;
  }

  // Writes an FST to a file; returns false on error; an empty filename
  // results in writing to standard output.
  virtual bool Write(const string &filename) const {
    LOG(ERROR) << "Fst::Write: No write filename method for " << Type()
               << " FST type";
    return false;
  }

  // Returns input label symbol table; return nullptr if not specified.
  virtual const SymbolTable *InputSymbols() const = 0;

  // Return output label symbol table; return nullptr if not specified.
  virtual const SymbolTable *OutputSymbols() const = 0;

  // For generic state iterator construction (not normally called directly by
  // users). Does not copy the FST.
  virtual void InitStateIterator(StateIteratorData<Arc> *data) const = 0;

  // For generic arc iterator construction (not normally called directly by
  // users). Does not copy the FST.
  virtual void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) const = 0;

  // For generic matcher construction (not normally called directly by users).
  // Does not copy the FST.
  virtual MatcherBase<Arc> *InitMatcher(MatchType match_type) const;

 protected:
  bool WriteFile(const string &filename) const {
    if (!filename.empty()) {
      std::ofstream strm(filename,
                               std::ios_base::out | std::ios_base::binary);
      if (!strm) {
        LOG(ERROR) << "Fst::Write: Can't open file: " << filename;
        return false;
      }
      bool val = Write(strm, FstWriteOptions(filename));
      if (!val) LOG(ERROR) << "Fst::Write failed: " << filename;
      return val;
    } else {
      return Write(std::cout, FstWriteOptions("standard output"));
    }
  }
};

// A useful alias when using StdArc.
using StdFst = Fst<StdArc>;

// State and arc iterator definitions.
//
// State iterator interface templated on the Arc definition; used for
// StateIterator specializations returned by the InitStateIterator FST method.
template <class Arc>
class StateIteratorBase {
 public:
  using StateId = typename Arc::StateId;

  virtual ~StateIteratorBase() {}

  // End of iterator?
  virtual bool Done() const = 0;
  // Returns current state (when !Done()).
  virtual StateId Value() const = 0;
  // Advances to next state (when !Done()).
  virtual void Next() = 0;
  // Resets to initial condition.
  virtual void Reset() = 0;
};

// StateIterator initialization data.

template <class Arc>
struct StateIteratorData {
  using StateId = typename Arc::StateId;

  // Specialized iterator if non-zero.
  StateIteratorBase<Arc> *base;
  // Otherwise, the total number of states.
  StateId nstates;

  StateIteratorData() : base(nullptr), nstates(0) {}

  StateIteratorData(const StateIteratorData &) = delete;
  StateIteratorData &operator=(const StateIteratorData &) = delete;
};

// Generic state iterator, templated on the FST definition (a wrapper
// around a pointer to a specific one). Here is a typical use:
//
//   for (StateIterator<StdFst> siter(fst);
//        !siter.Done();
//        siter.Next()) {
//     StateId s = siter.Value();
//     ...
//   }
// There is no copying of the FST.
template <class FST>
class StateIterator {
 public:
  using Arc = typename FST::Arc;
  using StateId = typename Arc::StateId;

  explicit StateIterator(const FST &fst) : s_(0) {
    fst.InitStateIterator(&data_);
  }

  ~StateIterator() { delete data_.base; }

  bool Done() const {
    return data_.base ? data_.base->Done() : s_ >= data_.nstates;
  }

  StateId Value() const { return data_.base ? data_.base->Value() : s_; }

  void Next() {
    if (data_.base) {
      data_.base->Next();
    } else {
      ++s_;
    }
  }

  void Reset() {
    if (data_.base) {
      data_.base->Reset();
    } else {
      s_ = 0;
    }
  }

 private:
  StateIteratorData<Arc> data_;
  StateId s_;
};

// Flags to control the behavior on an arc iterator.
static constexpr uint32 kArcILabelValue =
    0x0001;  // Value() gives valid ilabel.
static constexpr uint32 kArcOLabelValue = 0x0002;  //  "       "     " olabel.
static constexpr uint32 kArcWeightValue = 0x0004;  //  "       "     " weight.
static constexpr uint32 kArcNextStateValue =
    0x0008;                                    //  "       "     " nextstate.
static constexpr uint32 kArcNoCache = 0x0010;  // No need to cache arcs.

static constexpr uint32 kArcValueFlags =
    kArcILabelValue | kArcOLabelValue | kArcWeightValue | kArcNextStateValue;

static constexpr uint32 kArcFlags = kArcValueFlags | kArcNoCache;

// Arc iterator interface, templated on the arc definition; used for arc
// iterator specializations that are returned by the InitArcIterator FST method.
template <class Arc>
class ArcIteratorBase {
 public:
  using StateId = typename Arc::StateId;

  virtual ~ArcIteratorBase() {}

  // End of iterator?
  virtual bool Done() const = 0;
  // Returns current arc (when !Done()).
  virtual const Arc &Value() const = 0;
  // Advances to next arc (when !Done()).
  virtual void Next() = 0;
  // Returns current position.
  virtual size_t Position() const = 0;
  // Returns to initial condition.
  virtual void Reset() = 0;
  // Advances to arbitrary arc by position.
  virtual void Seek(size_t) = 0;
  // Returns current behavorial flags
  virtual uint32 Flags() const = 0;
  // Sets behavorial flags.
  virtual void SetFlags(uint32, uint32) = 0;
};

// ArcIterator initialization data.
template <class Arc>
struct ArcIteratorData {
  ArcIteratorData()
      : base(nullptr), arcs(nullptr), narcs(0), ref_count(nullptr) {}

  ArcIteratorData(const ArcIteratorData &) = delete;

  ArcIteratorData &operator=(const ArcIteratorData &) = delete;

  ArcIteratorBase<Arc> *base;  // Specialized iterator if non-zero.
  const Arc *arcs;             // O.w. arcs pointer
  size_t narcs;                // ... and arc count.
  int *ref_count;              // ... and reference count if non-zero.
};

// Generic arc iterator, templated on the FST definition (a wrapper around a
// pointer to a specific one). Here is a typical use:
//
//   for (ArcIterator<StdFst> aiter(fst, s);
//        !aiter.Done();
//         aiter.Next()) {
//     StdArc &arc = aiter.Value();
//     ...
//   }
// There is no copying of the FST.
template <class FST>
class ArcIterator {
 public:
  using Arc = typename FST::Arc;
  using StateId = typename Arc::StateId;

  ArcIterator(const FST &fst, StateId s) : i_(0) {
    fst.InitArcIterator(s, &data_);
  }

  explicit ArcIterator(const ArcIteratorData<Arc> &data) : data_(data), i_(0) {
    if (data_.ref_count) ++(*data_.ref_count);
  }

  ~ArcIterator() {
    if (data_.base) {
      delete data_.base;
    } else if (data_.ref_count) {
      --(*data_.ref_count);
    }
  }

  bool Done() const {
    return data_.base ? data_.base->Done() : i_ >= data_.narcs;
  }

  const Arc &Value() const {
    return data_.base ? data_.base->Value() : data_.arcs[i_];
  }

  void Next() {
    if (data_.base) {
      data_.base->Next();
    } else {
      ++i_;
    }
  }

  void Reset() {
    if (data_.base) {
      data_.base->Reset();
    } else {
      i_ = 0;
    }
  }

  void Seek(size_t a) {
    if (data_.base) {
      data_.base->Seek(a);
    } else {
      i_ = a;
    }
  }

  size_t Position() const { return data_.base ? data_.base->Position() : i_; }

  uint32 Flags() const {
    if (data_.base) {
      return data_.base->Flags();
    } else {
      return kArcValueFlags;
    }
  }

  void SetFlags(uint32 flags, uint32 mask) {
    if (data_.base) data_.base->SetFlags(flags, mask);
  }

 private:
  ArcIteratorData<Arc> data_;
  size_t i_;
};

}  // namespace fst

// ArcIterator placement operator new and destroy function; new needs to be in
// the global namespace.

template <class FST>
void *operator new(size_t size,
                   fst::MemoryPool<fst::ArcIterator<FST>> *pool) {
  return pool->Allocate();
}

namespace fst {

template <class FST>
void Destroy(ArcIterator<FST> *aiter, MemoryPool<ArcIterator<FST>> *pool) {
  if (aiter) {
    aiter->~ArcIterator<FST>();
    pool->Free(aiter);
  }
}

// Matcher definitions.

template <class Arc>
MatcherBase<Arc> *Fst<Arc>::InitMatcher(MatchType match_type) const {
  return nullptr;  // One should just use the default matcher.
}

// FST accessors, useful in high-performance applications.

namespace internal {

// General case, requires non-abstract, 'final' methods. Use for inlining.

template <class F>
inline typename F::Arc::Weight Final(const F &fst, typename F::Arc::StateId s) {
  return fst.F::Final(s);
}

template <class F>
inline ssize_t NumArcs(const F &fst, typename F::Arc::StateId s) {
  return fst.F::NumArcs(s);
}

template <class F>
inline ssize_t NumInputEpsilons(const F &fst, typename F::Arc::StateId s) {
  return fst.F::NumInputEpsilons(s);
}

template <class F>
inline ssize_t NumOutputEpsilons(const F &fst, typename F::Arc::StateId s) {
  return fst.F::NumOutputEpsilons(s);
}

// Fst<Arc> case, abstract methods.

template <class Arc>
inline typename Arc::Weight Final(const Fst<Arc> &fst,
                                  typename Arc::StateId s) {
  return fst.Final(s);
}

template <class Arc>
inline size_t NumArcs(const Fst<Arc> &fst, typename Arc::StateId s) {
  return fst.NumArcs(s);
}

template <class Arc>
inline size_t NumInputEpsilons(const Fst<Arc> &fst, typename Arc::StateId s) {
  return fst.NumInputEpsilons(s);
}

template <class Arc>
inline size_t NumOutputEpsilons(const Fst<Arc> &fst, typename Arc::StateId s) {
  return fst.NumOutputEpsilons(s);
}

// FST implementation base.
//
// This is the recommended FST implementation base class. It will handle
// reference counts, property bits, type information and symbols.
//
// Users are discouraged, but not prohibited, from subclassing this outside the
// FST library.
template <class Arc>
class FstImpl {
 public:
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  FstImpl() : properties_(0), type_("null") {}

  FstImpl(const FstImpl<Arc> &impl)
      : properties_(impl.properties_),
        type_(impl.type_),
        isymbols_(impl.isymbols_ ? impl.isymbols_->Copy() : nullptr),
        osymbols_(impl.osymbols_ ? impl.osymbols_->Copy() : nullptr) {}

  virtual ~FstImpl() {}

  const string &Type() const { return type_; }

  void SetType(const string &type) { type_ = type; }

  virtual uint64 Properties() const { return properties_; }

  virtual uint64 Properties(uint64 mask) const { return properties_ & mask; }

  void SetProperties(uint64 props) {
    properties_ &= kError;  // kError can't be cleared.
    properties_ |= props;
  }

  void SetProperties(uint64 props, uint64 mask) {
    properties_ &= ~mask | kError;  // kError can't be cleared.
    properties_ |= props & mask;
  }

  // Allows (only) setting error bit on const FST implementations.
  void SetProperties(uint64 props, uint64 mask) const {
    if (mask != kError) {
      FSTERROR() << "FstImpl::SetProperties() const: Can only set kError";
    }
    properties_ |= kError;
  }

  const SymbolTable *InputSymbols() const { return isymbols_.get(); }

  const SymbolTable *OutputSymbols() const { return osymbols_.get(); }

  SymbolTable *InputSymbols() { return isymbols_.get(); }

  SymbolTable *OutputSymbols() { return osymbols_.get(); }

  void SetInputSymbols(const SymbolTable *isyms) {
    isymbols_.reset(isyms ? isyms->Copy() : nullptr);
  }

  void SetOutputSymbols(const SymbolTable *osyms) {
    osymbols_.reset(osyms ? osyms->Copy() : nullptr);
  }

  // Reads header and symbols from input stream, initializes FST, and returns
  // the header. If opts.header is non-null, skips reading and uses the option
  // value instead. If opts.[io]symbols is non-null, reads in (if present), but
  // uses the option value.
  bool ReadHeader(std::istream &strm, const FstReadOptions &opts,
                  int min_version, FstHeader *hdr);

  // Writes header and symbols to output stream. If opts.header is false, skips
  // writing header. If opts.[io]symbols is false, skips writing those symbols.
  // This method is needed for implementations that implement Write methods.
  void WriteHeader(std::ostream &strm, const FstWriteOptions &opts,
                   int version, FstHeader *hdr) const {
    if (opts.write_header) {
      hdr->SetFstType(type_);
      hdr->SetArcType(Arc::Type());
      hdr->SetVersion(version);
      hdr->SetProperties(properties_);
      int32 file_flags = 0;
      if (isymbols_ && opts.write_isymbols) {
        file_flags |= FstHeader::HAS_ISYMBOLS;
      }
      if (osymbols_ && opts.write_osymbols) {
        file_flags |= FstHeader::HAS_OSYMBOLS;
      }
      if (opts.align) file_flags |= FstHeader::IS_ALIGNED;
      hdr->SetFlags(file_flags);
      hdr->Write(strm, opts.source);
    }
    if (isymbols_ && opts.write_isymbols) isymbols_->Write(strm);
    if (osymbols_ && opts.write_osymbols) osymbols_->Write(strm);
  }

  // Writes out header and symbols to output stream. If opts.header is false,
  // skips writing header. If opts.[io]symbols is false, skips writing those
  // symbols. `type` is the FST type being written. This method is used in the
  // cross-type serialization methods Fst::WriteFst.
  static void WriteFstHeader(const Fst<Arc> &fst, std::ostream &strm,
                             const FstWriteOptions &opts, int version,
                             const string &type, uint64 properties,
                             FstHeader *hdr) {
    if (opts.write_header) {
      hdr->SetFstType(type);
      hdr->SetArcType(Arc::Type());
      hdr->SetVersion(version);
      hdr->SetProperties(properties);
      int32 file_flags = 0;
      if (fst.InputSymbols() && opts.write_isymbols) {
        file_flags |= FstHeader::HAS_ISYMBOLS;
      }
      if (fst.OutputSymbols() && opts.write_osymbols) {
        file_flags |= FstHeader::HAS_OSYMBOLS;
      }
      if (opts.align) file_flags |= FstHeader::IS_ALIGNED;
      hdr->SetFlags(file_flags);
      hdr->Write(strm, opts.source);
    }
    if (fst.InputSymbols() && opts.write_isymbols) {
      fst.InputSymbols()->Write(strm);
    }
    if (fst.OutputSymbols() && opts.write_osymbols) {
      fst.OutputSymbols()->Write(strm);
    }
  }

  // In serialization routines where the header cannot be written until after
  // the machine has been serialized, this routine can be called to seek to the
  // beginning of the file an rewrite the header with updated fields. It
  // repositions the file pointer back at the end of the file. Returns true on
  // success, false on failure.
  static bool UpdateFstHeader(const Fst<Arc> &fst, std::ostream &strm,
                              const FstWriteOptions &opts, int version,
                              const string &type, uint64 properties,
                              FstHeader *hdr, size_t header_offset) {
    strm.seekp(header_offset);
    if (!strm) {
      LOG(ERROR) << "Fst::UpdateFstHeader: Write failed: " << opts.source;
      return false;
    }
    WriteFstHeader(fst, strm, opts, version, type, properties, hdr);
    if (!strm) {
      LOG(ERROR) << "Fst::UpdateFstHeader: Write failed: " << opts.source;
      return false;
    }
    strm.seekp(0, std::ios_base::end);
    if (!strm) {
      LOG(ERROR) << "Fst::UpdateFstHeader: Write failed: " << opts.source;
      return false;
    }
    return true;
  }

 protected:
  mutable uint64 properties_;  // Property bits.

 private:
  string type_;  // Unique name of FST class.
  std::unique_ptr<SymbolTable> isymbols_;
  std::unique_ptr<SymbolTable> osymbols_;
};

template <class Arc>
bool FstImpl<Arc>::ReadHeader(std::istream &strm, const FstReadOptions &opts,
                              int min_version, FstHeader *hdr) {
  if (opts.header) {
    *hdr = *opts.header;
  } else if (!hdr->Read(strm, opts.source)) {
    return false;
  }
  if (FLAGS_v >= 2) {
    LOG(INFO) << "FstImpl::ReadHeader: source: " << opts.source
              << ", fst_type: " << hdr->FstType()
              << ", arc_type: " << Arc::Type()
              << ", version: " << hdr->Version()
              << ", flags: " << hdr->GetFlags();
  }
  if (hdr->FstType() != type_) {
    LOG(ERROR) << "FstImpl::ReadHeader: FST not of type " << type_
               << ": " << opts.source;
    return false;
  }
  if (hdr->ArcType() != Arc::Type()) {
    LOG(ERROR) << "FstImpl::ReadHeader: Arc not of type " << Arc::Type()
               << ": " << opts.source;
    return false;
  }
  if (hdr->Version() < min_version) {
    LOG(ERROR) << "FstImpl::ReadHeader: Obsolete " << type_
               << " FST version: " << opts.source;
    return false;
  }
  properties_ = hdr->Properties();
  if (hdr->GetFlags() & FstHeader::HAS_ISYMBOLS) {
    isymbols_.reset(SymbolTable::Read(strm, opts.source));
  }
  // Deletes input symbol table.
  if (!opts.read_isymbols) SetInputSymbols(nullptr);
  if (hdr->GetFlags() & FstHeader::HAS_OSYMBOLS) {
    osymbols_.reset(SymbolTable::Read(strm, opts.source));
  }
  // Deletes output symbol table.
  if (!opts.read_osymbols) SetOutputSymbols(nullptr);
  if (opts.isymbols) {
    isymbols_.reset(opts.isymbols->Copy());
  }
  if (opts.osymbols) {
    osymbols_.reset(opts.osymbols->Copy());
  }
  return true;
}

}  // namespace internal

template <class Arc>
uint64 TestProperties(const Fst<Arc> &fst, uint64 mask, uint64 *known);

// This is a helper class template useful for attaching an FST interface to
// its implementation, handling reference counting.
template <class Impl, class FST = Fst<typename Impl::Arc>>
class ImplToFst : public FST {
 public:
  using Arc = typename Impl::Arc;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using FST::operator=;

  StateId Start() const override { return impl_->Start(); }

  Weight Final(StateId s) const override { return impl_->Final(s); }

  size_t NumArcs(StateId s) const override { return impl_->NumArcs(s); }

  size_t NumInputEpsilons(StateId s) const override {
    return impl_->NumInputEpsilons(s);
  }

  size_t NumOutputEpsilons(StateId s) const override {
    return impl_->NumOutputEpsilons(s);
  }

  uint64 Properties(uint64 mask, bool test) const override {
    if (test) {
      uint64 knownprops, testprops = TestProperties(*this, mask, &knownprops);
      impl_->SetProperties(testprops, knownprops);
      return testprops & mask;
    } else {
      return impl_->Properties(mask);
    }
  }

  const string &Type() const override { return impl_->Type(); }

  const SymbolTable *InputSymbols() const override {
    return impl_->InputSymbols();
  }

  const SymbolTable *OutputSymbols() const override {
    return impl_->OutputSymbols();
  }

 protected:
  explicit ImplToFst(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

  // This constructor presumes there is a copy constructor for the
  // implementation.
  ImplToFst(const ImplToFst<Impl, FST> &fst, bool safe) {
    if (safe) {
      impl_ = std::make_shared<Impl>(*(fst.impl_));
    } else {
      impl_ = fst.impl_;
    }
  }

  // Returns raw pointers to the shared object.
  const Impl *GetImpl() const { return impl_.get(); }

  Impl *GetMutableImpl() const { return impl_.get(); }

  // Returns a ref-counted smart poiner to the implementation.
  std::shared_ptr<Impl> GetSharedImpl() const { return impl_; }

  bool Unique() const { return impl_.unique(); }

  void SetImpl(std::shared_ptr<Impl> impl) { impl_ = impl; }

 private:
  template <class IFST, class OFST>
  friend void Cast(const IFST &ifst, OFST *ofst);

  std::shared_ptr<Impl> impl_;
};

// Converts FSTs by casting their implementations, where this makes sense
// (which excludes implementations with weight-dependent virtual methods).
// Must be a friend of the FST classes involved (currently the concrete FSTs:
// ConstFst, CompactFst, and VectorFst). This can only be safely used for arc
// types that have identical storage characteristics. As with an FST
// copy constructor and Copy() method, this is a constant time operation
// (but subject to copy-on-write if it is a MutableFst and modified).
template <class IFST, class OFST>
void Cast(const IFST &ifst, OFST *ofst) {
  using OImpl = typename OFST::Impl;
  ofst->impl_ = std::shared_ptr<OImpl>(ifst.impl_,
      reinterpret_cast<OImpl *>(ifst.impl_.get()));
}

// FST serialization.

template <class Arc>
string FstToString(const Fst<Arc> &fst,
                   const FstWriteOptions &options =
                       FstWriteOptions("FstToString")) {
  std::ostringstream ostrm;
  fst.Write(ostrm, options);
  return ostrm.str();
}

template <class Arc>
void FstToString(const Fst<Arc> &fst, string *result) {
  *result = FstToString(fst);
}

template <class Arc>
void FstToString(const Fst<Arc> &fst, string *result,
                 const FstWriteOptions &options) {
  *result = FstToString(fst, options);
}

template <class Arc>
Fst<Arc> *StringToFst(const string &s) {
  std::istringstream istrm(s);
  return Fst<Arc>::Read(istrm, FstReadOptions("StringToFst"));
}

}  // namespace fst

#endif  // FST_FST_H_
