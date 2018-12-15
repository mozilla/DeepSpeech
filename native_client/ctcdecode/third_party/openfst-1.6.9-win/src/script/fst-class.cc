// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// These classes are only recommended for use in high-level scripting
// applications. Most users should use the lower-level templated versions
// corresponding to these classes.

#include <istream>

#include <fst/log.h>

#include <fst/equal.h>
#include <fst/fst-decl.h>
#include <fst/reverse.h>
#include <fst/union.h>
#include <fst/script/fst-class.h>
#include <fst/script/register.h>

namespace fst {
namespace script {

// Registration.

REGISTER_FST_CLASSES(StdArc);
REGISTER_FST_CLASSES(LogArc);
REGISTER_FST_CLASSES(Log64Arc);

// Helper functions.

namespace {

template <class F>
F *ReadFstClass(std::istream &istrm, const string &fname) {
  if (!istrm) {
    LOG(ERROR) << "ReadFstClass: Can't open file: " << fname;
    return nullptr;
  }
  FstHeader hdr;
  if (!hdr.Read(istrm, fname)) return nullptr;
  const FstReadOptions read_options(fname, &hdr);
  const auto &arc_type = hdr.ArcType();
  static const auto *io_register = IORegistration<F>::Register::GetRegister();
  const auto reader = io_register->GetReader(arc_type);
  if (!reader) {
    LOG(ERROR) << "ReadFstClass: Unknown arc type: " << arc_type;
    return nullptr;
  }
  return reader(istrm, read_options);
}

template <class F>
FstClassImplBase *CreateFstClass(const string &arc_type) {
  static const auto *io_register =
      IORegistration<F>::Register::GetRegister();
  auto creator = io_register->GetCreator(arc_type);
  if (!creator) {
    FSTERROR() << "CreateFstClass: Unknown arc type: " << arc_type;
    return nullptr;
  }
  return creator();
}

template <class F>
FstClassImplBase *ConvertFstClass(const FstClass &other) {
  static const auto *io_register =
      IORegistration<F>::Register::GetRegister();
  auto converter = io_register->GetConverter(other.ArcType());
  if (!converter) {
    FSTERROR() << "ConvertFstClass: Unknown arc type: " << other.ArcType();
    return nullptr;
  }
  return converter(other);
}

}  // namespace


// FstClass methods.

FstClass *FstClass::Read(const string &fname) {
  if (!fname.empty()) {
    std::ifstream istrm(fname, std::ios_base::in | std::ios_base::binary);
    return ReadFstClass<FstClass>(istrm, fname);
  } else {
    return ReadFstClass<FstClass>(std::cin, "standard input");
  }
}

FstClass *FstClass::Read(std::istream &istrm, const string &source) {
  return ReadFstClass<FstClass>(istrm, source);
}

bool FstClass::WeightTypesMatch(const WeightClass &weight,
                                const string &op_name) const {
  if (WeightType() != weight.Type()) {
    FSTERROR() << "FST and weight with non-matching weight types passed to "
               << op_name << ": " << WeightType() << " and " << weight.Type();
    return false;
  }
  return true;
}

// MutableFstClass methods.

MutableFstClass *MutableFstClass::Read(const string &fname, bool convert) {
  if (convert == false) {
    if (!fname.empty()) {
      std::ifstream in(fname, std::ios_base::in | std::ios_base::binary);
      return ReadFstClass<MutableFstClass>(in, fname);
    } else {
      return ReadFstClass<MutableFstClass>(std::cin, "standard input");
    }
  } else {  // Converts to VectorFstClass if not mutable.
    std::unique_ptr<FstClass> ifst(FstClass::Read(fname));
    if (!ifst) return nullptr;
    if (ifst->Properties(kMutable, false) == kMutable) {
      return static_cast<MutableFstClass *>(ifst.release());
    } else {
      return new VectorFstClass(*ifst.release());
    }
  }
}

// VectorFstClass methods.

VectorFstClass *VectorFstClass::Read(const string &fname) {
  if (!fname.empty()) {
    std::ifstream in(fname, std::ios_base::in | std::ios_base::binary);
    return ReadFstClass<VectorFstClass>(in, fname);
  } else {
    return ReadFstClass<VectorFstClass>(std::cin, "standard input");
  }
}

VectorFstClass::VectorFstClass(const string &arc_type)
    : MutableFstClass(CreateFstClass<VectorFstClass>(arc_type)) {}


VectorFstClass::VectorFstClass(const FstClass &other)
    : MutableFstClass(ConvertFstClass<VectorFstClass>(other)) {}

}  // namespace script
}  // namespace fst
