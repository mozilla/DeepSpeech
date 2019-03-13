// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Generic FST augmented with state count-interface class definition.

#ifndef FST_EXPANDED_FST_H_
#define FST_EXPANDED_FST_H_

#include <sys/types.h>
#include <istream>
#include <string>

#include <fst/log.h>
#include <fstream>

#include <fst/fst.h>


namespace fst {

// A generic FST plus state count.
template <class A>
class ExpandedFst : public Fst<A> {
 public:
  using Arc = A;
  using StateId = typename Arc::StateId;

  virtual StateId NumStates() const = 0;  // State count

  // Get a copy of this ExpandedFst. See Fst<>::Copy() for further doc.
  ExpandedFst<Arc> *Copy(bool safe = false) const override = 0;

  // Read an ExpandedFst from an input stream; return NULL on error.
  static ExpandedFst<Arc> *Read(std::istream &strm,
                                const FstReadOptions &opts) {
    FstReadOptions ropts(opts);
    FstHeader hdr;
    if (ropts.header) {
      hdr = *opts.header;
    } else {
      if (!hdr.Read(strm, opts.source)) return nullptr;
      ropts.header = &hdr;
    }
    if (!(hdr.Properties() & kExpanded)) {
      LOG(ERROR) << "ExpandedFst::Read: Not an ExpandedFst: " << ropts.source;
      return nullptr;
    }
    const auto reader =
        FstRegister<Arc>::GetRegister()->GetReader(hdr.FstType());
    if (!reader) {
      LOG(ERROR) << "ExpandedFst::Read: Unknown FST type \"" << hdr.FstType()
                 << "\" (arc type = \"" << A::Type() << "\"): " << ropts.source;
      return nullptr;
    }
    auto *fst = reader(strm, ropts);
    if (!fst) return nullptr;
    return static_cast<ExpandedFst<Arc> *>(fst);
  }

  // Read an ExpandedFst from a file; return NULL on error.
  // Empty filename reads from standard input.
  static ExpandedFst<Arc> *Read(const string &filename) {
    if (!filename.empty()) {
      std::ifstream strm(filename,
                              std::ios_base::in | std::ios_base::binary);
      if (!strm) {
        LOG(ERROR) << "ExpandedFst::Read: Can't open file: " << filename;
        return nullptr;
      }
      return Read(strm, FstReadOptions(filename));
    } else {
      return Read(std::cin, FstReadOptions("standard input"));
    }
  }
};

namespace internal {

//  ExpandedFst<A> case - abstract methods.
template <class Arc>
inline typename Arc::Weight Final(const ExpandedFst<Arc> &fst,
                                  typename Arc::StateId s) {
  return fst.Final(s);
}

template <class Arc>
inline std::ptrdiff_t NumArcs(const ExpandedFst<Arc> &fst, typename Arc::StateId s) {
  return fst.NumArcs(s);
}

template <class Arc>
inline std::ptrdiff_t NumInputEpsilons(const ExpandedFst<Arc> &fst,
                                typename Arc::StateId s) {
  return fst.NumInputEpsilons(s);
}

template <class Arc>
inline std::ptrdiff_t NumOutputEpsilons(const ExpandedFst<Arc> &fst,
                                 typename Arc::StateId s) {
  return fst.NumOutputEpsilons(s);
}

}  // namespace internal

// A useful alias when using StdArc.
using StdExpandedFst = ExpandedFst<StdArc>;

// This is a helper class template useful for attaching an ExpandedFst
// interface to its implementation, handling reference counting. It
// delegates to ImplToFst the handling of the Fst interface methods.
template <class Impl, class FST = ExpandedFst<typename Impl::Arc>>
class ImplToExpandedFst : public ImplToFst<Impl, FST> {
 public:
  using Arc = typename FST::Arc;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  using ImplToFst<Impl, FST>::operator=;

  StateId NumStates() const override { return GetImpl()->NumStates(); }

 protected:
  using ImplToFst<Impl, FST>::GetImpl;

  explicit ImplToExpandedFst(std::shared_ptr<Impl> impl)
      : ImplToFst<Impl, FST>(impl) {}

  ImplToExpandedFst(const ImplToExpandedFst<Impl, FST> &fst)
      : ImplToFst<Impl, FST>(fst) {}

  ImplToExpandedFst(const ImplToExpandedFst<Impl, FST> &fst, bool safe)
      : ImplToFst<Impl, FST>(fst, safe) {}

  static Impl *Read(std::istream &strm, const FstReadOptions &opts) {
    return Impl::Read(strm, opts);
  }

  // Read FST implementation from a file; return NULL on error.
  // Empty filename reads from standard input.
  static Impl *Read(const string &filename) {
    if (!filename.empty()) {
      std::ifstream strm(filename,
                              std::ios_base::in | std::ios_base::binary);
      if (!strm) {
        LOG(ERROR) << "ExpandedFst::Read: Can't open file: " << filename;
        return nullptr;
      }
      return Impl::Read(strm, FstReadOptions(filename));
    } else {
      return Impl::Read(std::cin, FstReadOptions("standard input"));
    }
  }
};

// Function to return the number of states in an FST, counting them
// if necessary.
template <class Arc>
typename Arc::StateId CountStates(const Fst<Arc> &fst) {
  if (fst.Properties(kExpanded, false)) {
    const auto *efst = static_cast<const ExpandedFst<Arc> *>(&fst);
    return efst->NumStates();
  } else {
    typename Arc::StateId nstates = 0;
    for (StateIterator<Fst<Arc>> siter(fst); !siter.Done(); siter.Next()) {
      ++nstates;
    }
    return nstates;
  }
}

// Function to return the number of arcs in an FST.
template <class Arc>
typename Arc::StateId CountArcs(const Fst<Arc> &fst) {
  size_t narcs = 0;
  for (StateIterator<Fst<Arc>> siter(fst); !siter.Done(); siter.Next()) {
    narcs += fst.NumArcs(siter.Value());
  }
  return narcs;
}

}  // namespace fst

#endif  // FST_EXPANDED_FST_H_
