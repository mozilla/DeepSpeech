#ifndef UTIL_PROBING_HASH_TABLE_H
#define UTIL_PROBING_HASH_TABLE_H

#include "util/exception.hh"
#include "util/mmap.hh"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <vector>

#include <cassert>
#include <stdint.h>

namespace util {

/* Thrown when table grows too large */
class ProbingSizeException : public Exception {
  public:
    ProbingSizeException() throw() {}
    ~ProbingSizeException() throw() {}
};

// std::identity is an SGI extension :-(
struct IdentityHash {
  template <class T> T operator()(T arg) const { return arg; }
};

class DivMod {
  public:
    explicit DivMod(std::size_t buckets) : buckets_(buckets) {}

    static uint64_t RoundBuckets(uint64_t from) {
      return from;
    }

    template <class It> It Ideal(It begin, uint64_t hash) const {
      return begin + (hash % buckets_);
    }

    template <class BaseIt, class OutIt> void Next(BaseIt begin, BaseIt end, OutIt &it) const {
      if (++it == end) it = begin;
    }

    void Double() {
      buckets_ *= 2;
    }

  private:
    std::size_t buckets_;
};

class Power2Mod {
  public:
    explicit Power2Mod(std::size_t buckets) {
      UTIL_THROW_IF(!buckets || (((buckets - 1) & buckets)), ProbingSizeException, "Size " << buckets << " is not a power of 2.");
      mask_ = buckets - 1;
    }

    // Round up to next power of 2.
    static uint64_t RoundBuckets(uint64_t from) {
      --from;
      from |= from >> 1;
      from |= from >> 2;
      from |= from >> 4;
      from |= from >> 8;
      from |= from >> 16;
      from |= from >> 32;
      return from + 1;
    }

    template <class It> It Ideal(It begin, uint64_t hash) const {
      return begin + (hash & mask_);
    }

    template <class BaseIt, class OutIt> void Next(BaseIt begin, BaseIt /*end*/, OutIt &it) const {
      it = begin + ((it - begin + 1) & mask_);
    }

    void Double() {
      mask_ = (mask_ << 1) | 1;
    }

  private:
    std::size_t mask_;
};

template <class EntryT, class HashT, class EqualT> class AutoProbing;

/* Non-standard hash table
 * Buckets must be set at the beginning and must be greater than maximum number
 * of elements, else it throws ProbingSizeException.
 * Memory management and initialization is externalized to make it easier to
 * serialize these to disk and load them quickly.
 * Uses linear probing to find value.
 * Only insert and lookup operations.
 */
template <class EntryT, class HashT, class EqualT = std::equal_to<typename EntryT::Key>, class ModT = DivMod> class ProbingHashTable {
  public:
    typedef EntryT Entry;
    typedef typename Entry::Key Key;
    typedef const Entry *ConstIterator;
    typedef Entry *MutableIterator;
    typedef HashT Hash;
    typedef EqualT Equal;
    typedef ModT Mod;

    static uint64_t Size(uint64_t entries, float multiplier) {
      uint64_t buckets = Mod::RoundBuckets(std::max(entries + 1, static_cast<uint64_t>(multiplier * static_cast<float>(entries))));
      return buckets * sizeof(Entry);
    }

    // Must be assigned to later.
    ProbingHashTable() : mod_(1), entries_(0)
#ifdef DEBUG
      , initialized_(false)
#endif
    {}

    ProbingHashTable(void *start, std::size_t allocated, const Key &invalid = Key(), const Hash &hash_func = Hash(), const Equal &equal_func = Equal())
      : begin_(reinterpret_cast<MutableIterator>(start)),
        end_(begin_ + allocated / sizeof(Entry)),
        buckets_(end_ - begin_),
        invalid_(invalid),
        hash_(hash_func),
        equal_(equal_func),
        mod_(end_ - begin_),
        entries_(0)
#ifdef DEBUG
        , initialized_(true)
#endif
    {}

    void Relocate(void *new_base) {
      begin_ = reinterpret_cast<MutableIterator>(new_base);
      end_ = begin_ + buckets_;
    }

    MutableIterator Ideal(const Key key) {
      return mod_.Ideal(begin_, hash_(key));
    }
    ConstIterator Ideal(const Key key) const {
      return mod_.Ideal(begin_, hash_(key));
    }

    template <class T> MutableIterator Insert(const T &t) {
#ifdef DEBUG
      assert(initialized_);
#endif
      UTIL_THROW_IF(++entries_ >= buckets_, ProbingSizeException, "Hash table with " << buckets_ << " buckets is full.");
      return UncheckedInsert(t);
    }

    // Return true if the value was found (and not inserted).  This is consistent with Find but the opposite of hash_map!
    template <class T> bool FindOrInsert(const T &t, MutableIterator &out) {
#ifdef DEBUG
      assert(initialized_);
#endif
      for (MutableIterator i = Ideal(t.GetKey());;mod_.Next(begin_, end_, i)) {
        Key got(i->GetKey());
        if (equal_(got, t.GetKey())) { out = i; return true; }
        if (equal_(got, invalid_)) {
          UTIL_THROW_IF(++entries_ >= buckets_, ProbingSizeException, "Hash table with " << buckets_ << " buckets is full.");
          *i = t;
          out = i;
          return false;
        }
      }
    }

    void FinishedInserting() {}

    // Don't change anything related to GetKey,
    template <class Key> bool UnsafeMutableFind(const Key key, MutableIterator &out) {
#ifdef DEBUG
      assert(initialized_);
#endif
      for (MutableIterator i(Ideal(key));; mod_.Next(begin_, end_, i)) {
        Key got(i->GetKey());
        if (equal_(got, key)) { out = i; return true; }
        if (equal_(got, invalid_)) return false;
      }
    }

    // Like UnsafeMutableFind, but the key must be there.
    template <class Key> MutableIterator UnsafeMutableMustFind(const Key key) {
      for (MutableIterator i(Ideal(key));; mod_.Next(begin_, end_, i)) {
        Key got(i->GetKey());
        if (equal_(got, key)) { return i; }
        assert(!equal_(got, invalid_));
      }
    }

    // Iterator is both input and output.
    template <class Key> bool FindFromIdeal(const Key key, ConstIterator &i) const {
#ifdef DEBUG
      assert(initialized_);
#endif
      for (;; mod_.Next(begin_, end_, i)) {
        Key got(i->GetKey());
        if (equal_(got, key)) return true;
        if (equal_(got, invalid_)) return false;
      }
    }

    template <class Key> bool Find(const Key key, ConstIterator &out) const {
      out = Ideal(key);
      return FindFromIdeal(key, out);
    }

    // Like Find but we're sure it must be there.
    template <class Key> ConstIterator MustFind(const Key key) const {
      for (ConstIterator i(Ideal(key));; mod_.Next(begin_, end_, i)) {
        Key got(i->GetKey());
        if (equal_(got, key)) { return i; }
        assert(!equal_(got, invalid_));
      }
    }

    void Clear() {
      Entry invalid;
      invalid.SetKey(invalid_);
      std::fill(begin_, end_, invalid);
      entries_ = 0;
    }

    // Return number of entries assuming no serialization went on.
    std::size_t SizeNoSerialization() const {
      return entries_;
    }

    // Return memory size expected by Double.
    std::size_t DoubleTo() const {
      return buckets_ * 2 * sizeof(Entry);
    }

    // Inform the table that it has double the amount of memory.
    // Pass clear_new = false if you are sure the new memory is initialized
    // properly (to invalid_) i.e. by mremap.
    void Double(void *new_base, bool clear_new = true) {
      begin_ = static_cast<MutableIterator>(new_base);
      MutableIterator old_end = begin_ + buckets_;
      buckets_ *= 2;
      end_ = begin_ + buckets_;
      mod_.Double();
      if (clear_new) {
        Entry invalid;
        invalid.SetKey(invalid_);
        std::fill(old_end, end_, invalid);
      }
      std::vector<Entry> rolled_over;
      // Move roll-over entries to a buffer because they might not roll over anymore.  This should be small.
      for (MutableIterator i = begin_; i != old_end && !equal_(i->GetKey(), invalid_); ++i) {
        rolled_over.push_back(*i);
        i->SetKey(invalid_);
      }
      /* Re-insert everything.  Entries might go backwards to take over a
       * recently opened gap, stay, move to new territory, or wrap around.   If
       * an entry wraps around, it might go to a pointer greater than i (which
       * can happen at the beginning) and it will be revisited to possibly fill
       * in a gap created later.
       */
      Entry temp;
      for (MutableIterator i = begin_; i != old_end; ++i) {
        if (!equal_(i->GetKey(), invalid_)) {
          temp = *i;
          i->SetKey(invalid_);
          UncheckedInsert(temp);
        }
      }
      // Put the roll-over entries back in.
      for (typename std::vector<Entry>::const_iterator i(rolled_over.begin()); i != rolled_over.end(); ++i) {
        UncheckedInsert(*i);
      }
    }

    // Mostly for tests, check consistency of every entry.
    void CheckConsistency() {
      MutableIterator last;
      for (last = end_ - 1; last >= begin_ && !equal_(last->GetKey(), invalid_); --last) {}
      UTIL_THROW_IF(last == begin_, ProbingSizeException, "Completely full");
      MutableIterator i;
      // Beginning can be wrap-arounds.
      for (i = begin_; !equal_(i->GetKey(), invalid_); ++i) {
        MutableIterator ideal = Ideal(i->GetKey());
        UTIL_THROW_IF(ideal > i && ideal <= last, Exception, "Inconsistency at position " << (i - begin_) << " should be at " << (ideal - begin_));
      }
      MutableIterator pre_gap = i;
      for (; i != end_; ++i) {
        if (equal_(i->GetKey(), invalid_)) {
          pre_gap = i;
          continue;
        }
        MutableIterator ideal = Ideal(i->GetKey());
        UTIL_THROW_IF(ideal > i || ideal <= pre_gap, Exception, "Inconsistency at position " << (i - begin_) << " with ideal " << (ideal - begin_));
      }
    }

    ConstIterator RawBegin() const {
      return begin_;
    }
    ConstIterator RawEnd() const {
      return end_;
    }

  private:
    friend class AutoProbing<Entry, Hash, Equal>;

    template <class T> MutableIterator UncheckedInsert(const T &t) {
      for (MutableIterator i(Ideal(t.GetKey()));; mod_.Next(begin_, end_, i)) {
        if (equal_(i->GetKey(), invalid_)) { *i = t; return i; }
      }
    }

    MutableIterator begin_;
    MutableIterator end_;
    std::size_t buckets_;
    Key invalid_;
    Hash hash_;
    Equal equal_;
    Mod mod_;

    std::size_t entries_;
#ifdef DEBUG
    bool initialized_;
#endif
};

// Resizable linear probing hash table.  This owns the memory.
template <class EntryT, class HashT, class EqualT = std::equal_to<typename EntryT::Key> > class AutoProbing {
  private:
    typedef ProbingHashTable<EntryT, HashT, EqualT, Power2Mod> Backend;
  public:
    static std::size_t MemUsage(std::size_t size, float multiplier = 1.5) {
      return Backend::Size(size, multiplier);
    }

    typedef EntryT Entry;
    typedef typename Entry::Key Key;
    typedef const Entry *ConstIterator;
    typedef Entry *MutableIterator;
    typedef HashT Hash;
    typedef EqualT Equal;

    AutoProbing(std::size_t initial_size = 5, const Key &invalid = Key(), const Hash &hash_func = Hash(), const Equal &equal_func = Equal()) :
      allocated_(Backend::Size(initial_size, 1.2)), mem_(allocated_, KeyIsRawZero(invalid)), backend_(mem_.get(), allocated_, invalid, hash_func, equal_func) {
      threshold_ = std::min<std::size_t>(backend_.buckets_ - 1, backend_.buckets_ * 0.9);
      if (!KeyIsRawZero(invalid)) {
        Clear();
      }
    }

    // Assumes that the key is unique.  Multiple insertions won't cause a failure, just inconsistent lookup.
    template <class T> MutableIterator Insert(const T &t) {
      ++backend_.entries_;
      DoubleIfNeeded();
      return backend_.UncheckedInsert(t);
    }

    template <class T> bool FindOrInsert(const T &t, MutableIterator &out) {
      DoubleIfNeeded();
      return backend_.FindOrInsert(t, out);
    }

    template <class Key> bool UnsafeMutableFind(const Key key, MutableIterator &out) {
      return backend_.UnsafeMutableFind(key, out);
    }

    template <class Key> MutableIterator UnsafeMutableMustFind(const Key key) {
      return backend_.UnsafeMutableMustFind(key);
    }

    template <class Key> bool Find(const Key key, ConstIterator &out) const {
      return backend_.Find(key, out);
    }

    template <class Key> ConstIterator MustFind(const Key key) const {
      return backend_.MustFind(key);
    }

    std::size_t Size() const {
      return backend_.SizeNoSerialization();
    }

    void Clear() {
      backend_.Clear();
    }

    ConstIterator RawBegin() const {
      return backend_.RawBegin();
    }
    ConstIterator RawEnd() const {
      return backend_.RawEnd();
    }

  private:
    void DoubleIfNeeded() {
      if (UTIL_LIKELY(Size() < threshold_))
        return;
      HugeRealloc(backend_.DoubleTo(), KeyIsRawZero(backend_.invalid_), mem_);
      allocated_ = backend_.DoubleTo();
      backend_.Double(mem_.get(), !KeyIsRawZero(backend_.invalid_));
      threshold_ = std::min<std::size_t>(backend_.buckets_ - 1, backend_.buckets_ * 0.9);
    }

    bool KeyIsRawZero(const Key &key) {
      for (const uint8_t *i = reinterpret_cast<const uint8_t*>(&key); i < reinterpret_cast<const uint8_t*>(&key) + sizeof(Key); ++i) {
        if (*i) return false;
      }
      return true;
    }

    std::size_t allocated_;
    util::scoped_memory mem_;
    Backend backend_;
    std::size_t threshold_;
};

} // namespace util

#endif // UTIL_PROBING_HASH_TABLE_H
