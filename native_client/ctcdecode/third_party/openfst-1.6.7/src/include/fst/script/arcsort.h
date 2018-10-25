// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_ARCSORT_H_
#define FST_SCRIPT_ARCSORT_H_

#include <utility>

#include <fst/arcsort.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

enum ArcSortType {
  ILABEL_SORT,
  OLABEL_SORT
};

using ArcSortArgs = std::pair<MutableFstClass *, ArcSortType>;

template <class Arc>
void ArcSort(ArcSortArgs *args) {
  MutableFst<Arc> *fst = std::get<0>(*args)->GetMutableFst<Arc>();
  switch (std::get<1>(*args)) {
    case ILABEL_SORT: {
      const ILabelCompare<Arc> icomp;
      ArcSort(fst, icomp);
      return;
    }
    case OLABEL_SORT: {
      const OLabelCompare<Arc> ocomp;
      ArcSort(fst, ocomp);
      return;
    }
  }
}

void ArcSort(MutableFstClass *ofst, ArcSortType);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_ARCSORT_H_
