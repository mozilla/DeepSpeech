// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Class to represent and operate on sets of intervals.

#ifndef FST_INTERVAL_SET_H_
#define FST_INTERVAL_SET_H_

#include <algorithm>
#include <iostream>
#include <vector>


#include <fst/util.h>


namespace fst {

// Half-open integral interval [a, b) of signed integers of type T.
template <class T>
struct IntInterval {
  T begin;
  T end;

  IntInterval() : begin(-1), end(-1) {}

  IntInterval(T begin, T end) : begin(begin), end(end) {}

  bool operator<(const IntInterval<T> &i) const {
    return begin < i.begin || (begin == i.begin && end > i.end);
  }

  bool operator==(const IntInterval<T> &i) const {
    return begin == i.begin && end == i.end;
  }

  bool operator!=(const IntInterval<T> &i) const {
    return begin != i.begin || end != i.end;
  }

  std::istream &Read(std::istream &strm) {
    T n;
    ReadType(strm, &n);
    begin = n;
    ReadType(strm, &n);
    end = n;
    return strm;
  }

  std::ostream &Write(std::ostream &strm) const {
    T n = begin;
    WriteType(strm, n);
    n = end;
    WriteType(strm, n);
    return strm;
  }
};

// Stores IntIntervals<T> in a vector. In addition, keeps the count of points in
// all intervals.
template <class T>
class VectorIntervalStore {
 public:
  using Interval = IntInterval<T>;
  using Iterator = typename std::vector<Interval>::const_iterator;

  VectorIntervalStore() : count_(-1) {}

  std::vector<Interval> *MutableIntervals() { return &intervals_; }

  const Interval *Intervals() const { return intervals_.data(); }

  T Size() const { return intervals_.size(); }

  T Count() const { return count_; }

  void SetCount(T count) { count_ = count; }

  void Clear() {
    intervals_.clear();
    count_ = 0;
  }

  Iterator begin() const { return intervals_.begin(); }

  Iterator end() const { return intervals_.end(); }

  std::istream &Read(std::istream &strm) {
    ReadType(strm, &intervals_);
    return ReadType(strm, &count_);
  }

  std::ostream &Write(std::ostream &strm) const {
    WriteType(strm, intervals_);
    return WriteType(strm, count_);
  }

 private:
  std::vector<Interval> intervals_;
  T count_;
};

// Stores and operates on a set of half-open integral intervals [a, b)
// of signed integers of type T.
template <class T, class Store = VectorIntervalStore<T>>
class IntervalSet {
 public:
  using Interval = IntInterval<T>;

  template <class... A>
  explicit IntervalSet(A... args) : intervals_(args...) {}

  // Returns the interval set as a vector.
  std::vector<Interval> *MutableIntervals() {
    return intervals_.MutableIntervals();
  }

  // Returns a pointer to an array of Size() elements.
  const Interval *Intervals() const { return intervals_.Intervals(); }

  bool Empty() const { return Size() == 0; }

  T Size() const { return intervals_.Size(); }

  // Number of points in the intervals (undefined if not normalized).
  T Count() const { return intervals_.Count(); }

  void Clear() { intervals_.Clear(); }

  // Adds an interval set to the set. The result may not be normalized.
  void Union(const IntervalSet<T, Store> &iset) {
    intervals_.MutableIntervals()->insert(intervals_.MutableIntervals()->end(),
                                          iset.intervals_.begin(),
                                          iset.intervals_.end());
  }

  // Requires intervals be normalized.
  bool Member(T value) const {
    const Interval interval(value, value);
    auto lb = std::lower_bound(intervals_.begin(), intervals_.end(), interval);
    if (lb == intervals_.begin()) return false;
    return (--lb)->end > value;
  }

  // Requires intervals be normalized.
  bool operator==(const IntervalSet<T, Store> &iset) const {
    return Size() == iset.Size() &&
           std::equal(intervals_.begin(), intervals_.end(),
                      iset.intervals_.begin());
  }

  // Requires intervals be normalized.
  bool operator!=(const IntervalSet<T, Store> &iset) const {
    return Size() != iset.Size() ||
           !std::equal(intervals_.begin(), intervals_.end(),
                       iset.intervals_.begin());
  }

  bool Singleton() const {
    return Size() == 1 &&
           intervals_.begin()->begin + 1 == intervals_.begin()->end;
  }

  // Sorts, collapses overlapping and adjacent interals, and sets count.
  void Normalize();

  // Intersects an interval set with the set. Requires intervals be normalized.
  // The result is normalized.
  void Intersect(const IntervalSet<T, Store> &iset,
                 IntervalSet<T, Store> *oset) const;

  // Complements the set w.r.t [0, maxval). Requires intervals be normalized.
  // The result is normalized.
  void Complement(T maxval, IntervalSet<T, Store> *oset) const;

  // Subtract an interval set from the set. Requires intervals be normalized.
  // The result is normalized.
  void Difference(const IntervalSet<T, Store> &iset,
                  IntervalSet<T, Store> *oset) const;

  // Determines if an interval set overlaps with the set. Requires intervals be
  // normalized.
  bool Overlaps(const IntervalSet<T, Store> &iset) const;

  // Determines if an interval set overlaps with the set but neither is
  // contained in the other. Requires intervals be normalized.
  bool StrictlyOverlaps(const IntervalSet<T, Store> &iset) const;

  // Determines if an interval set is contained within the set. Requires
  // intervals be normalized.
  bool Contains(const IntervalSet<T, Store> &iset) const;

  std::istream &Read(std::istream &strm) { return intervals_.Read(strm); }

  std::ostream &Write(std::ostream &strm) const {
    return intervals_.Write(strm);
  }

  typename Store::Iterator begin() const { return intervals_.begin(); }

  typename Store::Iterator end() const { return intervals_.end(); }

 private:
  Store intervals_;
};

// Sorts, collapses overlapping and adjacent intervals, and sets count.
template <typename T, class Store>
void IntervalSet<T, Store>::Normalize() {
  auto &intervals = *intervals_.MutableIntervals();
  std::sort(intervals.begin(), intervals.end());
  T count = 0;
  T size = 0;
  for (T i = 0; i < intervals.size(); ++i) {
    auto &inti = intervals[i];
    if (inti.begin == inti.end) continue;
    for (T j = i + 1; j < intervals.size(); ++j) {
      auto &intj = intervals[j];
      if (intj.begin > inti.end) break;
      if (intj.end > inti.end) inti.end = intj.end;
      ++i;
    }
    count += inti.end - inti.begin;
    intervals[size++] = inti;
  }
  intervals.resize(size);
  intervals_.SetCount(count);
}

// Intersects an interval set with the set. Requires intervals be normalized.
// The result is normalized.
template <typename T, class Store>
void IntervalSet<T, Store>::Intersect(const IntervalSet<T, Store> &iset,
                                      IntervalSet<T, Store> *oset) const {
  auto *ointervals = oset->MutableIntervals();
  auto it1 = intervals_.begin();
  auto it2 = iset.intervals_.begin();
  ointervals->clear();
  T count = 0;
  while (it1 != intervals_.end() && it2 != iset.intervals_.end()) {
    if (it1->end <= it2->begin) {
      ++it1;
    } else if (it2->end <= it1->begin) {
      ++it2;
    } else {
      ointervals->emplace_back(std::max(it1->begin, it2->begin),
                               std::min(it1->end, it2->end));
      count += ointervals->back().end - ointervals->back().begin;
      if (it1->end < it2->end) {
        ++it1;
      } else {
        ++it2;
      }
    }
  }
  oset->intervals_.SetCount(count);
}

// Complements the set w.r.t [0, maxval). Requires intervals be normalized.
// The result is normalized.
template <typename T, class Store>
void IntervalSet<T, Store>::Complement(T maxval,
                                       IntervalSet<T, Store> *oset) const {
  auto *ointervals = oset->MutableIntervals();
  ointervals->clear();
  T count = 0;
  Interval interval;
  interval.begin = 0;
  for (auto it = intervals_.begin(); it != intervals_.end(); ++it) {
    interval.end = std::min(it->begin, maxval);
    if ((interval.begin) < (interval.end)) {
      ointervals->push_back(interval);
      count += interval.end - interval.begin;
    }
    interval.begin = it->end;
  }
  interval.end = maxval;
  if ((interval.begin) < (interval.end)) {
    ointervals->push_back(interval);
    count += interval.end - interval.begin;
  }
  oset->intervals_.SetCount(count);
}

// Subtract an interval set from the set. Requires intervals be normalized.
// The result is normalized.
template <typename T, class Store>
void IntervalSet<T, Store>::Difference(const IntervalSet<T, Store> &iset,
                                       IntervalSet<T, Store> *oset) const {
  if (Empty()) {
    oset->MutableIntervals()->clear();
    oset->intervals_.SetCount(0);
  } else {
    IntervalSet<T, Store> cset;
    iset.Complement(intervals_.Intervals()[intervals_.Size() - 1].end, &cset);
    Intersect(cset, oset);
  }
}

// Determines if an interval set overlaps with the set. Requires intervals be
// normalized.
template <typename T, class Store>
bool IntervalSet<T, Store>::Overlaps(const IntervalSet<T, Store> &iset) const {
  auto it1 = intervals_.begin();
  auto it2 = iset.intervals_.begin();
  while (it1 != intervals_.end() && it2 != iset.intervals_.end()) {
    if (it1->end <= it2->begin) {
      ++it1;
    } else if (it2->end <= it1->begin) {
      ++it2;
    } else {
      return true;
    }
  }
  return false;
}

// Determines if an interval set overlaps with the set but neither is contained
// in the other. Requires intervals be normalized.
template <typename T, class Store>
bool IntervalSet<T, Store>::StrictlyOverlaps(
    const IntervalSet<T, Store> &iset) const {
  auto it1 = intervals_.begin();
  auto it2 = iset.intervals_.begin();
  bool only1 = false;    // Point in intervals_ but not intervals.
  bool only2 = false;    // Point in intervals but not intervals_.
  bool overlap = false;  // Point in both intervals_ and intervals.
  while (it1 != intervals_.end() && it2 != iset.intervals_.end()) {
    if (it1->end <= it2->begin) {  // no overlap - it1 first
      only1 = true;
      ++it1;
    } else if (it2->end <= it1->begin) {  // no overlap - it2 first
      only2 = true;
      ++it2;
    } else if (it2->begin == it1->begin && it2->end == it1->end) {  // equals
      overlap = true;
      ++it1;
      ++it2;
    } else if (it2->begin <= it1->begin && it2->end >= it1->end) {  // 1 c 2
      only2 = true;
      overlap = true;
      ++it1;
    } else if (it1->begin <= it2->begin && it1->end >= it2->end) {  // 2 c 1
      only1 = true;
      overlap = true;
      ++it2;
    } else {  // Strict overlap.
      only1 = true;
      only2 = true;
      overlap = true;
    }
    if (only1 == true && only2 == true && overlap == true) return true;
  }
  if (it1 != intervals_.end()) only1 = true;
  if (it2 != iset.intervals_.end()) only2 = true;
  return only1 == true && only2 == true && overlap == true;
}

// Determines if an interval set is contained within the set. Requires intervals
// be normalized.
template <typename T, class Store>
bool IntervalSet<T, Store>::Contains(const IntervalSet<T, Store> &iset) const {
  if (iset.Count() > Count()) return false;
  auto it1 = intervals_.begin();
  auto it2 = iset.intervals_.begin();
  while (it1 != intervals_.end() && it2 != iset.intervals_.end()) {
    if ((it1->end) <= (it2->begin)) {  // No overlap; it1 first.
      ++it1;
    } else if ((it2->begin) < (it1->begin) ||
               (it2->end) > (it1->end)) {  // No C.
      return false;
    } else if (it2->end == it1->end) {
      ++it1;
      ++it2;
    } else {
      ++it2;
    }
  }
  return it2 == iset.intervals_.end();
}

template <typename T, class Store>
std::ostream &operator<<(std::ostream &strm, const IntervalSet<T, Store> &s) {
  strm << "{";
  for (T i = 0; i < s.Size(); ++i) {
    if (i > 0) {
      strm << ",";
    }
    const auto &interval = s.Intervals()[i];
    strm << "[" << interval.begin << "," << interval.end << ")";
  }
  strm << "}";
  return strm;
}

}  // namespace fst

#endif  // FST_INTERVAL_SET_H_
