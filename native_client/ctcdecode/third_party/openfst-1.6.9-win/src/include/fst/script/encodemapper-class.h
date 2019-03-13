// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_ENCODEMAPPER_CLASS_H_
#define FST_SCRIPT_ENCODEMAPPER_CLASS_H_

#include <memory>
#include <string>
#include <iostream>

#include <fst/fstlib.h>
#include <fst/script/arc-class.h>
#include <fst/script/fst-class.h>

// Scripting API support for EncodeMapper.

namespace fst {
namespace script {

// Virtual interface implemented by each concrete EncodeMapperClassImpl<A>.
class EncodeMapperImplBase {
 public:
  // Returns an encoded ArcClass.
  virtual ArcClass operator()(const ArcClass &a) = 0;
  virtual const string &ArcType() const = 0;
  virtual uint32_t Flags() const = 0;
  virtual uint64_t Properties(uint64_t inprops) = 0;
  virtual EncodeType Type() const = 0;
  virtual const SymbolTable *InputSymbols() const = 0;
  virtual const SymbolTable *OutputSymbols() const = 0;
  virtual void SetInputSymbols(const SymbolTable *syms) = 0;
  virtual void SetOutputSymbols(const SymbolTable *syms) = 0;
  virtual const string &WeightType() const = 0;
  virtual ~EncodeMapperImplBase() {}
};

// Templated implementation.
template <class Arc>
class EncodeMapperClassImpl : public EncodeMapperImplBase {
 public:
  EncodeMapperClassImpl(uint32_t flags, EncodeType type)
      : encoder_(flags, type) {}

  ArcClass operator()(const ArcClass &a) final;

  const string &ArcType() const final { return Arc::Type(); }

  uint32_t Flags() const final { return encoder_.Flags(); }

  uint64_t Properties(uint64_t inprops) final {
    return encoder_.Properties(inprops);
  }

  EncodeType Type() const final { return encoder_.Type(); }

  const SymbolTable *InputSymbols() const final {
    return encoder_.InputSymbols();
  }

  const SymbolTable *OutputSymbols() const final {
    return encoder_.OutputSymbols();
  }

  void SetInputSymbols(const SymbolTable *syms) final {
    encoder_.SetInputSymbols(syms);
  }

  void SetOutputSymbols(const SymbolTable *syms) final {
    encoder_.SetOutputSymbols(syms);
  }

  const string &WeightType() const final { return Arc::Weight::Type(); }

  ~EncodeMapperClassImpl() override {}

  EncodeMapper<Arc> *GetImpl() const { return &encoder_; }

  EncodeMapper<Arc> *GetImpl() { return &encoder_; }

 private:
  EncodeMapper<Arc> encoder_;
};

// This is returned by value because it is very likely to undergo return-value
// optimization.
template <class Arc>
inline ArcClass EncodeMapperClassImpl<Arc>::operator()(const ArcClass &a) {
  Arc arc(a.ilabel, a.olabel, *(a.weight.GetWeight<typename Arc::Weight>()),
          a.nextstate);
  return ArcClass(encoder_(arc));
}

class EncodeMapperClass;

using InitEncodeMapperClassArgs =
    std::tuple<uint32_t, EncodeType, EncodeMapperClass *>;

class EncodeMapperClass {
 public:
  EncodeMapperClass(const string &arc_type, uint32_t flags, EncodeType type);

  template <class Arc>
  EncodeMapperClass(uint32_t flags, EncodeType type)
      : impl_(new EncodeMapperClassImpl<Arc>(flags, type)) {}

  ArcClass operator()(const ArcClass &arc) { return (*impl_)(arc); }

  const string &ArcType() const { return impl_->ArcType(); }

  uint32_t Flags() const { return impl_->Flags(); }

  uint64_t Properties(uint64_t inprops) { return impl_->Properties(inprops); }

  EncodeType Type() const { return impl_->Type(); }

  const SymbolTable *InputSymbols() const { return impl_->InputSymbols(); }

  const SymbolTable *OutputSymbols() const { return impl_->OutputSymbols(); }

  void SetInputSymbols(const SymbolTable *syms) {
    impl_->SetInputSymbols(syms);
  }

  void SetOutputSymbols(const SymbolTable *syms) {
    impl_->SetOutputSymbols(syms);
  }

  const string &WeightType() const { return impl_->WeightType(); }

  template <class Arc>
  friend void InitEncodeMapperClass(InitEncodeMapperClassArgs *args);

  // Naturally, this exists in non-const and const forms. Encoding arcs or FSTs
  // mutates the underlying encoder; decoding them does not.

  template <class Arc>
  EncodeMapper<Arc> *GetEncodeMapper() {
    if (Arc::Type() != ArcType()) {
      return nullptr;
    } else {
      auto *typed_impl = static_cast<EncodeMapperClassImpl<Arc> *>(impl_.get());
      return typed_impl->GetImpl();
    }
  }

  template <class Arc>
  const EncodeMapper<Arc> *GetEncodeMapper() const {
    if (Arc::Type() != ArcType()) {
      return nullptr;
    } else {
      auto *typed_impl = static_cast<EncodeMapperClassImpl<Arc> *>(impl_.get());
      return typed_impl->GetImpl();
    }
  }

 private:
  std::unique_ptr<EncodeMapperImplBase> impl_;
};

template <class Arc>
void InitEncodeMapperClass(InitEncodeMapperClassArgs *args) {
  std::get<2>(*args)->impl_.reset(
      new EncodeMapperClassImpl<Arc>(std::get<0>(*args), std::get<1>(*args)));
}

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_ENCODEMAPPER_CLASS_H_
