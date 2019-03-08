// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_STATEITERATOR_CLASS_H_
#define FST_SCRIPT_STATEITERATOR_CLASS_H_

#include <memory>

#include <fst/fstlib.h>
#include <fst/script/fst-class.h>

// Scripting API support for StateIterator.

namespace fst {
namespace script {

// Virtual interface implemented by each concrete StateIteratorImpl<F>.
class StateIteratorImplBase {
 public:
  virtual bool Done() const = 0;
  virtual int64_t Value() const = 0;
  virtual void Next() = 0;
  virtual void Reset() = 0;
  virtual ~StateIteratorImplBase() {}
};

// Templated implementation.
template <class Arc>
class StateIteratorClassImpl : public StateIteratorImplBase {
 public:
  explicit StateIteratorClassImpl(const Fst<Arc> &fst) : siter_(fst) {}

  bool Done() const final { return siter_.Done(); }

  int64_t Value() const final { return siter_.Value(); }

  void Next() final { siter_.Next(); }

  void Reset() final { siter_.Reset(); }

  ~StateIteratorClassImpl() override {}

 private:
  StateIterator<Fst<Arc>> siter_;
};

class StateIteratorClass;

using InitStateIteratorClassArgs =
    std::pair<const FstClass &, StateIteratorClass *>;

// Untemplated user-facing class holding a templated pimpl.
class StateIteratorClass {
 public:
  explicit StateIteratorClass(const FstClass &fst);

  template <class Arc>
  explicit StateIteratorClass(const Fst<Arc> &fst)
      : impl_(new StateIteratorClassImpl<Arc>(fst)) {}

  bool Done() const { return impl_->Done(); }

  int64_t Value() const { return impl_->Value(); }

  void Next() { impl_->Next(); }

  void Reset() { impl_->Reset(); }

  template <class Arc>
  friend void InitStateIteratorClass(InitStateIteratorClassArgs *args);

 private:
  std::unique_ptr<StateIteratorImplBase> impl_;
};

template <class Arc>
void InitStateIteratorClass(InitStateIteratorClassArgs *args) {
  const Fst<Arc> &fst = *(std::get<0>(*args).GetFst<Arc>());
  std::get<1>(*args)->impl_.reset(new StateIteratorClassImpl<Arc>(fst));
}

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_STATEITERATOR_CLASS_H_
