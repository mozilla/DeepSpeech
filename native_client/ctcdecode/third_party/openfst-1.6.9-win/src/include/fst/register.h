// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Classes for registering derived FST for generic reading.

#ifndef FST_REGISTER_H_
#define FST_REGISTER_H_

#include <string>
#include <type_traits>


#include <fst/compat.h>
#include <fst/generic-register.h>
#include <fst/util.h>


#include <fst/types.h>
#include <fst/log.h>

namespace fst {

template <class Arc>
class Fst;

struct FstReadOptions;

// This class represents a single entry in a FstRegister
template <class Arc>
struct FstRegisterEntry {
  using Reader = Fst<Arc> *(*)(std::istream &istrm, const FstReadOptions &opts);
  using Converter = Fst<Arc> *(*)(const Fst<Arc> &fst);

  Reader reader;
  Converter converter;

  explicit FstRegisterEntry(Reader reader = nullptr,
                            Converter converter = nullptr)
      : reader(reader), converter(converter) {}
};

// This class maintains the correspondence between a string describing
// an FST type, and its reader and converter.
template <class Arc>
class FstRegister
    : public GenericRegister<string, FstRegisterEntry<Arc>, FstRegister<Arc>> {
 public:
  using Reader = typename FstRegisterEntry<Arc>::Reader;
  using Converter = typename FstRegisterEntry<Arc>::Converter;

  const Reader GetReader(const string &type) const {
    return this->GetEntry(type).reader;
  }

  const Converter GetConverter(const string &type) const {
    return this->GetEntry(type).converter;
  }

 protected:
  string ConvertKeyToSoFilename(const string &key) const override {
    string legal_type(key);
    ConvertToLegalCSymbol(&legal_type);
    return legal_type + "-fst.so";
  }
};

// This class registers an FST type for generic reading and creating.
// The type must have a default constructor and a copy constructor from
// Fst<Arc>.
template <class FST>
class FstRegisterer : public GenericRegisterer<FstRegister<typename FST::Arc>> {
 public:
  using Arc = typename FST::Arc;
  using Entry = typename FstRegister<Arc>::Entry;
  using Reader = typename FstRegister<Arc>::Reader;

  FstRegisterer()
      : GenericRegisterer<FstRegister<typename FST::Arc>>(FST().Type(),
                                                          BuildEntry()) {}

 private:
  static Fst<Arc> *ReadGeneric(
      std::istream &strm, const FstReadOptions &opts) {
    static_assert(std::is_base_of<Fst<Arc>, FST>::value,
                  "FST class does not inherit from Fst<Arc>");
    return FST::Read(strm, opts);
  }

  static Entry BuildEntry() {
    return Entry(&ReadGeneric, &FstRegisterer<FST>::Convert);
  }

  static Fst<Arc> *Convert(const Fst<Arc> &fst) { return new FST(fst); }
};

// Convenience macro to generate static FstRegisterer instance.
#define REGISTER_FST(FST, Arc) \
  static fst::FstRegisterer<FST<Arc>> FST##_##Arc##_registerer

// Converts an FST to the specified type.
template <class Arc>
Fst<Arc> *Convert(const Fst<Arc> &fst, const string &fst_type) {
  auto *reg = FstRegister<Arc>::GetRegister();
  const auto converter = reg->GetConverter(fst_type);
  if (!converter) {
    FSTERROR() << "Fst::Convert: Unknown FST type " << fst_type << " (arc type "
               << Arc::Type() << ")";
    return nullptr;
  }
  return converter(fst);
}

}  // namespace fst

#endif  // FST_REGISTER_H_
