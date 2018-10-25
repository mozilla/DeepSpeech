// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Finite-State Transducer (FST) archive classes.

#ifndef FST_EXTENSIONS_FAR_FAR_H_
#define FST_EXTENSIONS_FAR_FAR_H_

#include <iostream>
#include <sstream>

#include <fst/log.h>
#include <fst/extensions/far/stlist.h>
#include <fst/extensions/far/sttable.h>
#include <fst/fst.h>
#include <fst/vector-fst.h>
#include <fstream>

namespace fst {

enum FarEntryType { FET_LINE, FET_FILE };

enum FarTokenType { FTT_SYMBOL, FTT_BYTE, FTT_UTF8 };

inline bool IsFst(const string &filename) {
  std::ifstream strm(filename, std::ios_base::in | std::ios_base::binary);
  if (!strm) return false;
  return IsFstHeader(strm, filename);
}

// FST archive header class
class FarHeader {
 public:
  const string &ArcType() const { return arctype_; }

  const string &FarType() const { return fartype_; }

  bool Read(const string &filename) {
    FstHeader fsthdr;
    if (filename.empty()) {
      // Header reading unsupported on stdin. Assumes STList and StdArc.
      fartype_ = "stlist";
      arctype_ = "standard";
      return true;
    } else if (IsSTTable(filename)) {  // Checks if STTable.
      ReadSTTableHeader(filename, &fsthdr);
      fartype_ = "sttable";
      arctype_ = fsthdr.ArcType().empty() ? "unknown" : fsthdr.ArcType();
      return true;
    } else if (IsSTList(filename)) {  // Checks if STList.
      ReadSTListHeader(filename, &fsthdr);
      fartype_ = "stlist";
      arctype_ = fsthdr.ArcType().empty() ? "unknown" : fsthdr.ArcType();
      return true;
    } else if (IsFst(filename)) {  // Checks if FST.
      std::ifstream istrm(filename,
                               std::ios_base::in | std::ios_base::binary);
      fsthdr.Read(istrm, filename);
      fartype_ = "fst";
      arctype_ = fsthdr.ArcType().empty() ? "unknown" : fsthdr.ArcType();
      return true;
    }
    return false;
  }

 private:
  string fartype_;
  string arctype_;
};

enum FarType {
  FAR_DEFAULT = 0,
  FAR_STTABLE = 1,
  FAR_STLIST = 2,
  FAR_FST = 3,
};

// This class creates an archive of FSTs.
template <class A>
class FarWriter {
 public:
  using Arc = A;

  // Creates a new (empty) FST archive; returns null on error.
  static FarWriter *Create(const string &filename, FarType type = FAR_DEFAULT);

  // Adds an FST to the end of an archive. Keys must be non-empty and
  // in lexicographic order. FSTs must have a suitable write method.
  virtual void Add(const string &key, const Fst<Arc> &fst) = 0;

  virtual FarType Type() const = 0;

  virtual bool Error() const = 0;

  virtual ~FarWriter() {}

 protected:
  FarWriter() {}
};

// This class iterates through an existing archive of FSTs.
template <class A>
class FarReader {
 public:
  using Arc = A;

  // Opens an existing FST archive in a single file; returns null on error.
  // Sets current position to the beginning of the achive.
  static FarReader *Open(const string &filename);

  // Opens an existing FST archive in multiple files; returns null on error.
  // Sets current position to the beginning of the achive.
  static FarReader *Open(const std::vector<string> &filenames);

  // Resets current position to beginning of archive.
  virtual void Reset() = 0;

  // Sets current position to first entry >= key.  Returns true if a match.
  virtual bool Find(const string &key) = 0;

  // Current position at end of archive?
  virtual bool Done() const = 0;

  // Move current position to next FST.
  virtual void Next() = 0;

  // Returns key at the current position. This reference is invalidated if
  // the current position in the archive is changed.
  virtual const string &GetKey() const = 0;

  // Returns pointer to FST at the current position. This is invalidated if
  // the current position in the archive is changed.
  virtual const Fst<Arc> *GetFst() const = 0;

  virtual FarType Type() const = 0;

  virtual bool Error() const = 0;

  virtual ~FarReader() {}

 protected:
  FarReader() {}
};

template <class Arc>
class FstWriter {
 public:
  void operator()(std::ostream &strm, const Fst<Arc> &fst) const {
    fst.Write(strm, FstWriteOptions());
  }
};

template <class A>
class STTableFarWriter : public FarWriter<A> {
 public:
  using Arc = A;

  static STTableFarWriter *Create(const string &filename) {
    auto *writer = STTableWriter<Fst<Arc>, FstWriter<Arc>>::Create(filename);
    return new STTableFarWriter(writer);
  }

  void Add(const string &key, const Fst<Arc> &fst) final {
    writer_->Add(key, fst);
  }

  FarType Type() const final { return FAR_STTABLE; }

  bool Error() const final { return writer_->Error(); }

 private:
  explicit STTableFarWriter(STTableWriter<Fst<Arc>, FstWriter<Arc>> *writer)
      : writer_(writer) {}

  std::unique_ptr<STTableWriter<Fst<Arc>, FstWriter<Arc>>> writer_;
};

template <class A>
class STListFarWriter : public FarWriter<A> {
 public:
  using Arc = A;

  static STListFarWriter *Create(const string &filename) {
    auto *writer = STListWriter<Fst<Arc>, FstWriter<Arc>>::Create(filename);
    return new STListFarWriter(writer);
  }

  void Add(const string &key, const Fst<Arc> &fst) final {
    writer_->Add(key, fst);
  }

  constexpr FarType Type() const final { return FAR_STLIST; }

  bool Error() const final { return writer_->Error(); }

 private:
  explicit STListFarWriter(STListWriter<Fst<Arc>, FstWriter<Arc>> *writer)
      : writer_(writer) {}

  std::unique_ptr<STListWriter<Fst<Arc>, FstWriter<Arc>>> writer_;
};

template <class A>
class FstFarWriter : public FarWriter<A> {
 public:
  using Arc = A;

  explicit FstFarWriter(const string &filename)
      : filename_(filename), error_(false), written_(false) {}

  static FstFarWriter *Create(const string &filename) {
    return new FstFarWriter(filename);
  }

  void Add(const string &key, const Fst<A> &fst) final {
    if (written_) {
      LOG(WARNING) << "FstFarWriter::Add: only one FST supported,"
                   << " subsequent entries discarded.";
    } else {
      error_ = !fst.Write(filename_);
      written_ = true;
    }
  }

  constexpr FarType Type() const final { return FAR_FST; }

  bool Error() const final { return error_; }

  ~FstFarWriter() final {}

 private:
  string filename_;
  bool error_;
  bool written_;
};

template <class Arc>
FarWriter<Arc> *FarWriter<Arc>::Create(const string &filename, FarType type) {
  switch (type) {
    case FAR_DEFAULT:
      if (filename.empty()) return STListFarWriter<Arc>::Create(filename);
    case FAR_STTABLE:
      return STTableFarWriter<Arc>::Create(filename);
    case FAR_STLIST:
      return STListFarWriter<Arc>::Create(filename);
    case FAR_FST:
      return FstFarWriter<Arc>::Create(filename);
    default:
      LOG(ERROR) << "FarWriter::Create: Unknown FAR type";
      return nullptr;
  }
}

template <class Arc>
class FstReader {
 public:
  Fst<Arc> *operator()(std::istream &strm) const {
    return Fst<Arc>::Read(strm, FstReadOptions());
  }
};

template <class A>
class STTableFarReader : public FarReader<A> {
 public:
  using Arc = A;

  static STTableFarReader *Open(const string &filename) {
    auto *reader = STTableReader<Fst<Arc>, FstReader<Arc>>::Open(filename);
    if (!reader || reader->Error()) return nullptr;
    return new STTableFarReader(reader);
  }

  static STTableFarReader *Open(const std::vector<string> &filenames) {
    auto *reader = STTableReader<Fst<Arc>, FstReader<Arc>>::Open(filenames);
    if (!reader || reader->Error()) return nullptr;
    return new STTableFarReader(reader);
  }

  void Reset() final { reader_->Reset(); }

  bool Find(const string &key) final { return reader_->Find(key); }

  bool Done() const final { return reader_->Done(); }

  void Next() final { return reader_->Next(); }

  const string &GetKey() const final { return reader_->GetKey(); }

  const Fst<Arc> *GetFst() const final { return reader_->GetEntry(); }

  constexpr FarType Type() const final { return FAR_STTABLE; }

  bool Error() const final { return reader_->Error(); }

 private:
  explicit STTableFarReader(STTableReader<Fst<Arc>, FstReader<Arc>> *reader)
      : reader_(reader) {}

  std::unique_ptr<STTableReader<Fst<Arc>, FstReader<Arc>>> reader_;
};

template <class A>
class STListFarReader : public FarReader<A> {
 public:
  using Arc = A;

  static STListFarReader *Open(const string &filename) {
    auto *reader = STListReader<Fst<Arc>, FstReader<Arc>>::Open(filename);
    if (!reader || reader->Error()) return nullptr;
    return new STListFarReader(reader);
  }

  static STListFarReader *Open(const std::vector<string> &filenames) {
    auto *reader = STListReader<Fst<Arc>, FstReader<Arc>>::Open(filenames);
    if (!reader || reader->Error()) return nullptr;
    return new STListFarReader(reader);
  }

  void Reset() final { reader_->Reset(); }

  bool Find(const string &key) final { return reader_->Find(key); }

  bool Done() const final { return reader_->Done(); }

  void Next() final { return reader_->Next(); }

  const string &GetKey() const final { return reader_->GetKey(); }

  const Fst<Arc> *GetFst() const final { return reader_->GetEntry(); }

  constexpr FarType Type() const final { return FAR_STLIST; }

  bool Error() const final { return reader_->Error(); }

 private:
  explicit STListFarReader(STListReader<Fst<Arc>, FstReader<Arc>> *reader)
      : reader_(reader) {}

  std::unique_ptr<STListReader<Fst<Arc>, FstReader<Arc>>> reader_;
};

template <class A>
class FstFarReader : public FarReader<A> {
 public:
  using Arc = A;

  static FstFarReader *Open(const string &filename) {
    std::vector<string> filenames;
    filenames.push_back(filename);
    return new FstFarReader<Arc>(filenames);
  }

  static FstFarReader *Open(const std::vector<string> &filenames) {
    return new FstFarReader<Arc>(filenames);
  }

  explicit FstFarReader(const std::vector<string> &filenames)
      : keys_(filenames), has_stdin_(false), pos_(0), error_(false) {
    std::sort(keys_.begin(), keys_.end());
    streams_.resize(keys_.size(), 0);
    for (size_t i = 0; i < keys_.size(); ++i) {
      if (keys_[i].empty()) {
        if (!has_stdin_) {
          streams_[i] = &std::cin;
          // sources_[i] = "stdin";
          has_stdin_ = true;
        } else {
          FSTERROR() << "FstFarReader::FstFarReader: standard input should "
                        "only appear once in the input file list";
          error_ = true;
          return;
        }
      } else {
        streams_[i] = new std::ifstream(
            keys_[i], std::ios_base::in | std::ios_base::binary);
      }
    }
    if (pos_ >= keys_.size()) return;
    ReadFst();
  }

  void Reset() final {
    if (has_stdin_) {
      FSTERROR()
          << "FstFarReader::Reset: Operation not supported on standard input";
      error_ = true;
      return;
    }
    pos_ = 0;
    ReadFst();
  }

  bool Find(const string &key) final {
    if (has_stdin_) {
      FSTERROR()
          << "FstFarReader::Find: Operation not supported on standard input";
      error_ = true;
      return false;
    }
    pos_ = 0;  // TODO
    ReadFst();
    return true;
  }

  bool Done() const final { return error_ || pos_ >= keys_.size(); }

  void Next() final {
    ++pos_;
    ReadFst();
  }

  const string &GetKey() const final { return keys_[pos_]; }

  const Fst<Arc> *GetFst() const final { return fst_.get(); }

  constexpr FarType Type() const final { return FAR_FST; }

  bool Error() const final { return error_; }

  ~FstFarReader() final {
    for (size_t i = 0; i < keys_.size(); ++i) {
      if (streams_[i] != &std::cin) {
        delete streams_[i];
      }
    }
  }

 private:
  void ReadFst() {
    fst_.reset();
    if (pos_ >= keys_.size()) return;
    streams_[pos_]->seekg(0);
    fst_.reset(Fst<Arc>::Read(*streams_[pos_], FstReadOptions()));
    if (!fst_) {
      FSTERROR() << "FstFarReader: Error reading Fst from: " << keys_[pos_];
      error_ = true;
    }
  }

  std::vector<string> keys_;
  std::vector<std::istream *> streams_;
  bool has_stdin_;
  size_t pos_;
  mutable std::unique_ptr<Fst<Arc>> fst_;
  mutable bool error_;
};

template <class Arc>
FarReader<Arc> *FarReader<Arc>::Open(const string &filename) {
  if (filename.empty())
    return STListFarReader<Arc>::Open(filename);
  else if (IsSTTable(filename))
    return STTableFarReader<Arc>::Open(filename);
  else if (IsSTList(filename))
    return STListFarReader<Arc>::Open(filename);
  else if (IsFst(filename))
    return FstFarReader<Arc>::Open(filename);
  return nullptr;
}

template <class Arc>
FarReader<Arc> *FarReader<Arc>::Open(const std::vector<string> &filenames) {
  if (!filenames.empty() && filenames[0].empty())
    return STListFarReader<Arc>::Open(filenames);
  else if (!filenames.empty() && IsSTTable(filenames[0]))
    return STTableFarReader<Arc>::Open(filenames);
  else if (!filenames.empty() && IsSTList(filenames[0]))
    return STListFarReader<Arc>::Open(filenames);
  else if (!filenames.empty() && IsFst(filenames[0]))
    return FstFarReader<Arc>::Open(filenames);
  return nullptr;
}

}  // namespace fst

#endif  // FST_EXTENSIONS_FAR_FAR_H_
