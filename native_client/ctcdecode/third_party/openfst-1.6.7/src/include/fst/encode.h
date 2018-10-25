// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Class to encode and decode an FST.

#ifndef FST_ENCODE_H_
#define FST_ENCODE_H_

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fst/log.h>
#include <fstream>

#include <fst/arc-map.h>
#include <fst/rmfinalepsilon.h>


namespace fst {

enum EncodeType { ENCODE = 1, DECODE = 2 };

static constexpr uint32 kEncodeLabels = 0x0001;
static constexpr uint32 kEncodeWeights = 0x0002;
static constexpr uint32 kEncodeFlags = 0x0003;

namespace internal {

static constexpr uint32 kEncodeHasISymbols = 0x0004;
static constexpr uint32 kEncodeHasOSymbols = 0x0008;

// Identifies stream data as an encode table (and its endianity)
static const int32 kEncodeMagicNumber = 2129983209;

// The following class encapsulates implementation details for the encoding and
// decoding of label/weight tuples used for encoding and decoding of FSTs. The
// EncodeTable is bidirectional. I.e, it stores both the Tuple of encode labels
// and weights to a unique label, and the reverse.
template <class Arc>
class EncodeTable {
 public:
  using Label = typename Arc::Label;
  using Weight = typename Arc::Weight;

  // Encoded data consists of arc input/output labels and arc weight.
  struct Tuple {
    Tuple() {}

    Tuple(Label ilabel_, Label olabel_, Weight weight_)
        : ilabel(ilabel_), olabel(olabel_), weight(std::move(weight_)) {}

    Tuple(const Tuple &tuple)
        : ilabel(tuple.ilabel),
          olabel(tuple.olabel),
          weight(std::move(tuple.weight)) {}

    Label ilabel;
    Label olabel;
    Weight weight;
  };

  // Comparison object for hashing EncodeTable Tuple(s).
  class TupleEqual {
   public:
    bool operator()(const Tuple *x, const Tuple *y) const {
      return (x->ilabel == y->ilabel && x->olabel == y->olabel &&
              x->weight == y->weight);
    }
  };

  // Hash function for EncodeTabe Tuples. Based on the encode flags
  // we either hash the labels, weights or combination of them.
  class TupleKey {
   public:
    TupleKey() : encode_flags_(kEncodeLabels | kEncodeWeights) {}

    TupleKey(const TupleKey &key) : encode_flags_(key.encode_flags_) {}

    explicit TupleKey(uint32 encode_flags) : encode_flags_(encode_flags) {}

    size_t operator()(const Tuple *x) const {
      size_t hash = x->ilabel;
      static constexpr int lshift = 5;
      static constexpr int rshift = CHAR_BIT * sizeof(size_t) - 5;
      if (encode_flags_ & kEncodeLabels) {
        hash = hash << lshift ^ hash >> rshift ^ x->olabel;
      }
      if (encode_flags_ & kEncodeWeights) {
        hash = hash << lshift ^ hash >> rshift ^ x->weight.Hash();
      }
      return hash;
    }

   private:
    int32 encode_flags_;
  };

  explicit EncodeTable(uint32 encode_flags)
      : flags_(encode_flags), encode_hash_(1024, TupleKey(encode_flags)) {}

  using EncodeHash = std::unordered_map<const Tuple *, Label, TupleKey,
                                        TupleEqual>;

  // Given an arc, encodes either input/output labels or input/costs or both.
  Label Encode(const Arc &arc) {
    std::unique_ptr<Tuple> tuple(
        new Tuple(arc.ilabel, flags_ & kEncodeLabels ? arc.olabel : 0,
                  flags_ & kEncodeWeights ? arc.weight : Weight::One()));
    auto insert_result = encode_hash_.insert(
        std::make_pair(tuple.get(), encode_tuples_.size() + 1));
    if (insert_result.second) encode_tuples_.push_back(std::move(tuple));
    return insert_result.first->second;
  }

  // Given an arc, looks up its encoded label or returns kNoLabel if not found.
  Label GetLabel(const Arc &arc) const {
    const Tuple tuple(arc.ilabel, flags_ & kEncodeLabels ? arc.olabel : 0,
                      flags_ & kEncodeWeights ? arc.weight : Weight::One());
    auto it = encode_hash_.find(&tuple);
    return (it == encode_hash_.end()) ?  kNoLabel : it->second;
  }

  // Given an encoded arc label, decodes back to input/output labels and costs.
  const Tuple *Decode(Label key) const {
    if (key < 1 || key > encode_tuples_.size()) {
      LOG(ERROR) << "EncodeTable::Decode: Unknown decode key: " << key;
      return nullptr;
    }
    return encode_tuples_[key - 1].get();
  }

  size_t Size() const { return encode_tuples_.size(); }

  bool Write(std::ostream &strm, const string &source) const;

  static EncodeTable<Arc> *Read(std::istream &strm, const string &source);

  uint32 Flags() const { return flags_ & kEncodeFlags; }

  const SymbolTable *InputSymbols() const { return isymbols_.get(); }

  const SymbolTable *OutputSymbols() const { return osymbols_.get(); }

  void SetInputSymbols(const SymbolTable *syms) {
    if (syms) {
      isymbols_.reset(syms->Copy());
      flags_ |= kEncodeHasISymbols;
    } else {
      isymbols_.reset();
      flags_ &= ~kEncodeHasISymbols;
    }
  }

  void SetOutputSymbols(const SymbolTable *syms) {
    if (syms) {
      osymbols_.reset(syms->Copy());
      flags_ |= kEncodeHasOSymbols;
    } else {
      osymbols_.reset();
      flags_ &= ~kEncodeHasOSymbols;
    }
  }

 private:
  uint32 flags_;
  std::vector<std::unique_ptr<Tuple>> encode_tuples_;
  EncodeHash encode_hash_;
  std::unique_ptr<SymbolTable> isymbols_;  // Pre-encoded input symbol table.
  std::unique_ptr<SymbolTable> osymbols_;  // Pre-encoded output symbol table.

  EncodeTable(const EncodeTable &) = delete;
  EncodeTable &operator=(const EncodeTable &) = delete;
};

template <class Arc>
bool EncodeTable<Arc>::Write(std::ostream &strm,
                                  const string &source) const {
  WriteType(strm, kEncodeMagicNumber);
  WriteType(strm, flags_);
  const int64 size = encode_tuples_.size();
  WriteType(strm, size);
  for (const auto &tuple : encode_tuples_) {
    WriteType(strm, tuple->ilabel);
    WriteType(strm, tuple->olabel);
    tuple->weight.Write(strm);
  }
  if (flags_ & kEncodeHasISymbols) isymbols_->Write(strm);
  if (flags_ & kEncodeHasOSymbols) osymbols_->Write(strm);
  strm.flush();
  if (!strm) {
    LOG(ERROR) << "EncodeTable::Write: Write failed: " << source;
    return false;
  }
  return true;
}

template <class Arc>
EncodeTable<Arc> *EncodeTable<Arc>::Read(std::istream &strm,
                                         const string &source) {
  int32 magic_number = 0;
  ReadType(strm, &magic_number);
  if (magic_number != kEncodeMagicNumber) {
    LOG(ERROR) << "EncodeTable::Read: Bad encode table header: " << source;
    return nullptr;
  }
  uint32 flags;
  ReadType(strm, &flags);
  int64 size;
  ReadType(strm, &size);
  if (!strm) {
    LOG(ERROR) << "EncodeTable::Read: Read failed: " << source;
    return nullptr;
  }
  std::unique_ptr<EncodeTable<Arc>> table(new EncodeTable<Arc>(flags));
  for (int64 i = 0; i < size; ++i) {
    std::unique_ptr<Tuple> tuple(new Tuple());
    ReadType(strm, &tuple->ilabel);
    ReadType(strm, &tuple->olabel);
    tuple->weight.Read(strm);
    if (!strm) {
      LOG(ERROR) << "EncodeTable::Read: Read failed: " << source;
      return nullptr;
    }
    table->encode_tuples_.push_back(std::move(tuple));
    table->encode_hash_[table->encode_tuples_.back().get()] =
        table->encode_tuples_.size();
  }
  if (flags & kEncodeHasISymbols) {
    table->isymbols_.reset(SymbolTable::Read(strm, source));
  }
  if (flags & kEncodeHasOSymbols) {
    table->osymbols_.reset(SymbolTable::Read(strm, source));
  }
  return table.release();
}

}  // namespace internal

// A mapper to encode/decode weighted transducers. Encoding of an FST is used
// for performing classical determinization or minimization on a weighted
// transducer viewing it as an unweighted acceptor over encoded labels.
//
// The mapper stores the encoding in a local hash table (EncodeTable). This
// table is shared (and reference-counted) between the encoder and decoder.
// A decoder has read-only access to the EncodeTable.
//
// The EncodeMapper allows on the fly encoding of the machine. As the
// EncodeTable is generated the same table may by used to decode the machine
// on the fly. For example in the following sequence of operations
//
//  Encode -> Determinize -> Decode
//
// we will use the encoding table generated during the encode step in the
// decode, even though the encoding is not complete.
template <class Arc>
class EncodeMapper {
  using Label = typename Arc::Label;
  using Weight = typename Arc::Weight;

 public:
  EncodeMapper(uint32 flags, EncodeType type)
      : flags_(flags),
        type_(type),
        table_(std::make_shared<internal::EncodeTable<Arc>>(flags)),
        error_(false) {}

  EncodeMapper(const EncodeMapper &mapper)
      : flags_(mapper.flags_),
        type_(mapper.type_),
        table_(mapper.table_),
        error_(false) {}

  // Copy constructor but setting the type, typically to DECODE.
  EncodeMapper(const EncodeMapper &mapper, EncodeType type)
      : flags_(mapper.flags_),
        type_(type),
        table_(mapper.table_),
        error_(mapper.error_) {}

  Arc operator()(const Arc &arc);

  MapFinalAction FinalAction() const {
    return (type_ == ENCODE && (flags_ & kEncodeWeights))
               ? MAP_REQUIRE_SUPERFINAL
               : MAP_NO_SUPERFINAL;
  }

  constexpr MapSymbolsAction InputSymbolsAction() const {
    return MAP_CLEAR_SYMBOLS;
  }

  constexpr MapSymbolsAction OutputSymbolsAction() const {
    return MAP_CLEAR_SYMBOLS;
  }

  uint64 Properties(uint64 inprops) {
    uint64 outprops = inprops;
    if (error_) outprops |= kError;
    uint64 mask = kFstProperties;
    if (flags_ & kEncodeLabels) {
      mask &= kILabelInvariantProperties & kOLabelInvariantProperties;
    }
    if (flags_ & kEncodeWeights) {
      mask &= kILabelInvariantProperties & kWeightInvariantProperties &
              (type_ == ENCODE ? kAddSuperFinalProperties
                               : kRmSuperFinalProperties);
    }
    return outprops & mask;
  }

  uint32 Flags() const { return flags_; }

  EncodeType Type() const { return type_; }

  bool Write(std::ostream &strm, const string &source) const {
    return table_->Write(strm, source);
  }

  bool Write(const string &filename) const {
    std::ofstream strm(filename,
                             std::ios_base::out | std::ios_base::binary);
    if (!strm) {
      LOG(ERROR) << "EncodeMap: Can't open file: " << filename;
      return false;
    }
    return Write(strm, filename);
  }

  static EncodeMapper<Arc> *Read(std::istream &strm, const string &source,
                               EncodeType type = ENCODE) {
    auto *table = internal::EncodeTable<Arc>::Read(strm, source);
    return table ? new EncodeMapper(table->Flags(), type, table) : nullptr;
  }

  static EncodeMapper<Arc> *Read(const string &filename,
                                 EncodeType type = ENCODE) {
    std::ifstream strm(filename,
                            std::ios_base::in | std::ios_base::binary);
    if (!strm) {
      LOG(ERROR) << "EncodeMap: Can't open file: " << filename;
      return nullptr;
    }
    return Read(strm, filename, type);
  }

  const SymbolTable *InputSymbols() const { return table_->InputSymbols(); }

  const SymbolTable *OutputSymbols() const { return table_->OutputSymbols(); }

  void SetInputSymbols(const SymbolTable *syms) {
    table_->SetInputSymbols(syms);
  }

  void SetOutputSymbols(const SymbolTable *syms) {
    table_->SetOutputSymbols(syms);
  }

 private:
  uint32 flags_;
  EncodeType type_;
  std::shared_ptr<internal::EncodeTable<Arc>> table_;
  bool error_;

  explicit EncodeMapper(uint32 flags, EncodeType type,
                        internal::EncodeTable<Arc> *table)
      : flags_(flags), type_(type), table_(table), error_(false) {}

  EncodeMapper &operator=(const EncodeMapper &) = delete;
};

template <class Arc>
Arc EncodeMapper<Arc>::operator()(const Arc &arc) {
  if (type_ == ENCODE) {
    if ((arc.nextstate == kNoStateId && !(flags_ & kEncodeWeights)) ||
        (arc.nextstate == kNoStateId && (flags_ & kEncodeWeights) &&
         arc.weight == Weight::Zero())) {
      return arc;
    } else {
      const auto label = table_->Encode(arc);
      return Arc(label, flags_ & kEncodeLabels ? label : arc.olabel,
                 flags_ & kEncodeWeights ? Weight::One() : arc.weight,
                 arc.nextstate);
    }
  } else {  // type_ == DECODE
    if (arc.nextstate == kNoStateId) {
      return arc;
    } else {
      if (arc.ilabel == 0) return arc;
      if (flags_ & kEncodeLabels && arc.ilabel != arc.olabel) {
        FSTERROR() << "EncodeMapper: Label-encoded arc has different "
                      "input and output labels";
        error_ = true;
      }
      if (flags_ & kEncodeWeights && arc.weight != Weight::One()) {
        FSTERROR() << "EncodeMapper: Weight-encoded arc has non-trivial weight";
        error_ = true;
      }
      const auto tuple = table_->Decode(arc.ilabel);
      if (!tuple) {
        FSTERROR() << "EncodeMapper: Decode failed";
        error_ = true;
        return Arc(kNoLabel, kNoLabel, Weight::NoWeight(), arc.nextstate);
      } else {
        return Arc(tuple->ilabel,
                   flags_ & kEncodeLabels ? tuple->olabel : arc.olabel,
                   flags_ & kEncodeWeights ? tuple->weight : arc.weight,
                   arc.nextstate);
      }
    }
  }
}

// Complexity: O(E + V).
template <class Arc>
inline void Encode(MutableFst<Arc> *fst, EncodeMapper<Arc> *mapper) {
  mapper->SetInputSymbols(fst->InputSymbols());
  mapper->SetOutputSymbols(fst->OutputSymbols());
  ArcMap(fst, mapper);
}

template <class Arc>
inline void Decode(MutableFst<Arc> *fst, const EncodeMapper<Arc> &mapper) {
  ArcMap(fst, EncodeMapper<Arc>(mapper, DECODE));
  RmFinalEpsilon(fst);
  fst->SetInputSymbols(mapper.InputSymbols());
  fst->SetOutputSymbols(mapper.OutputSymbols());
}

// On-the-fly encoding of an input FST.
//
// Complexity:
//
//   Construction: O(1)
//   Traversal: O(e + v)
//
// where e is the number of arcs visited and v is the number of states visited.
// Constant time and space to visit an input state or arc is assumed and
// exclusive of caching.
template <class Arc>
class EncodeFst : public ArcMapFst<Arc, Arc, EncodeMapper<Arc>> {
 public:
  using Mapper = EncodeMapper<Arc>;
  using Impl = internal::ArcMapFstImpl<Arc, Arc, Mapper>;

  EncodeFst(const Fst<Arc> &fst, Mapper *encoder)
      : ArcMapFst<Arc, Arc, Mapper>(fst, encoder, ArcMapFstOptions()) {
    encoder->SetInputSymbols(fst.InputSymbols());
    encoder->SetOutputSymbols(fst.OutputSymbols());
  }

  EncodeFst(const Fst<Arc> &fst, const Mapper &encoder)
      : ArcMapFst<Arc, Arc, Mapper>(fst, encoder, ArcMapFstOptions()) {}

  // See Fst<>::Copy() for doc.
  EncodeFst(const EncodeFst<Arc> &fst, bool copy = false)
      : ArcMapFst<Arc, Arc, Mapper>(fst, copy) {}

  // Makes a copy of this EncodeFst. See Fst<>::Copy() for further doc.
  EncodeFst<Arc> *Copy(bool safe = false) const override {
    if (safe) {
      FSTERROR() << "EncodeFst::Copy(true): Not allowed";
      GetImpl()->SetProperties(kError, kError);
    }
    return new EncodeFst(*this);
  }

 private:
  using ImplToFst<Impl>::GetImpl;
  using ImplToFst<Impl>::GetMutableImpl;
};

// On-the-fly decoding of an input FST.
//
// Complexity:
//
//   Construction: O(1).
//   Traversal: O(e + v)
//
// Constant time and space to visit an input state or arc is assumed and
// exclusive of caching.
template <class Arc>
class DecodeFst : public ArcMapFst<Arc, Arc, EncodeMapper<Arc>> {
 public:
  using Mapper = EncodeMapper<Arc>;
  using Impl = internal::ArcMapFstImpl<Arc, Arc, Mapper>;
  using ImplToFst<Impl>::GetImpl;

  DecodeFst(const Fst<Arc> &fst, const Mapper &encoder)
      : ArcMapFst<Arc, Arc, Mapper>(fst, Mapper(encoder, DECODE),
                                    ArcMapFstOptions()) {
    GetMutableImpl()->SetInputSymbols(encoder.InputSymbols());
    GetMutableImpl()->SetOutputSymbols(encoder.OutputSymbols());
  }

  // See Fst<>::Copy() for doc.
  DecodeFst(const DecodeFst<Arc> &fst, bool safe = false)
      : ArcMapFst<Arc, Arc, Mapper>(fst, safe) {}

  // Makes a copy of this DecodeFst. See Fst<>::Copy() for further doc.
  DecodeFst<Arc> *Copy(bool safe = false) const override {
    return new DecodeFst(*this, safe);
  }

 private:
  using ImplToFst<Impl>::GetMutableImpl;
};

// Specialization for EncodeFst.
template <class Arc>
class StateIterator<EncodeFst<Arc>>
    : public StateIterator<ArcMapFst<Arc, Arc, EncodeMapper<Arc>>> {
 public:
  explicit StateIterator(const EncodeFst<Arc> &fst)
      : StateIterator<ArcMapFst<Arc, Arc, EncodeMapper<Arc>>>(fst) {}
};

// Specialization for EncodeFst.
template <class Arc>
class ArcIterator<EncodeFst<Arc>>
    : public ArcIterator<ArcMapFst<Arc, Arc, EncodeMapper<Arc>>> {
 public:
  ArcIterator(const EncodeFst<Arc> &fst, typename Arc::StateId s)
      : ArcIterator<ArcMapFst<Arc, Arc, EncodeMapper<Arc>>>(fst, s) {}
};

// Specialization for DecodeFst.
template <class Arc>
class StateIterator<DecodeFst<Arc>>
    : public StateIterator<ArcMapFst<Arc, Arc, EncodeMapper<Arc>>> {
 public:
  explicit StateIterator(const DecodeFst<Arc> &fst)
      : StateIterator<ArcMapFst<Arc, Arc, EncodeMapper<Arc>>>(fst) {}
};

// Specialization for DecodeFst.
template <class Arc>
class ArcIterator<DecodeFst<Arc>>
    : public ArcIterator<ArcMapFst<Arc, Arc, EncodeMapper<Arc>>> {
 public:
  ArcIterator(const DecodeFst<Arc> &fst, typename Arc::StateId s)
      : ArcIterator<ArcMapFst<Arc, Arc, EncodeMapper<Arc>>>(fst, s) {}
};

// Useful aliases when using StdArc.

using StdEncodeFst = EncodeFst<StdArc>;

using StdDecodeFst = DecodeFst<StdArc>;

}  // namespace fst

#endif  // FST_ENCODE_H_
