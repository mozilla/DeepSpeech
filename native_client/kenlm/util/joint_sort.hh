#ifndef UTIL_JOINT_SORT_H
#define UTIL_JOINT_SORT_H

/* A terrifying amount of C++ to coax std::sort into soring one range while
 * also permuting another range the same way.
 */

#include "util/proxy_iterator.hh"

#include <algorithm>
#include <functional>

namespace util {

namespace detail {

template <class KeyIter, class ValueIter> class JointProxy;

template <class KeyIter, class ValueIter> class JointIter {
  public:
    JointIter() {}

    JointIter(const KeyIter &key_iter, const ValueIter &value_iter) : key_(key_iter), value_(value_iter) {}

    bool operator==(const JointIter<KeyIter, ValueIter> &other) const { return key_ == other.key_; }

    bool operator<(const JointIter<KeyIter, ValueIter> &other) const { return (key_ < other.key_); }

    std::ptrdiff_t operator-(const JointIter<KeyIter, ValueIter> &other) const { return key_ - other.key_; }

    JointIter<KeyIter, ValueIter> &operator+=(std::ptrdiff_t amount) {
      key_ += amount;
      value_ += amount;
      return *this;
    }

    friend void swap(JointIter &first, JointIter &second) {
      using std::swap;
      swap(first.key_, second.key_);
      swap(first.value_, second.value_);
    }

    void DeepSwap(JointIter &other) {
      using std::swap;
      swap(*key_, *other.key_);
      swap(*value_, *other.value_);
    }

  private:
    friend class JointProxy<KeyIter, ValueIter>;
    KeyIter key_;
    ValueIter value_;
};

template <class KeyIter, class ValueIter> class JointProxy {
  private:
    typedef JointIter<KeyIter, ValueIter> InnerIterator;

  public:
    typedef struct {
      typename std::iterator_traits<KeyIter>::value_type key;
      typename std::iterator_traits<ValueIter>::value_type value;
      const typename std::iterator_traits<KeyIter>::value_type &GetKey() const { return key; }
    } value_type;

    JointProxy(const KeyIter &key_iter, const ValueIter &value_iter) : inner_(key_iter, value_iter) {}
    JointProxy(const JointProxy<KeyIter, ValueIter> &other) : inner_(other.inner_) {}

    operator value_type() const {
      value_type ret;
      ret.key = *inner_.key_;
      ret.value = *inner_.value_;
      return ret;
    }

    JointProxy &operator=(const JointProxy &other) {
      *inner_.key_ = *other.inner_.key_;
      *inner_.value_ = *other.inner_.value_;
      return *this;
    }

    JointProxy &operator=(const value_type &other) {
      *inner_.key_ = other.key;
      *inner_.value_ = other.value;
      return *this;
    }

    typename std::iterator_traits<KeyIter>::reference GetKey() const {
      return *(inner_.key_);
    }

    friend void swap(JointProxy<KeyIter, ValueIter> first, JointProxy<KeyIter, ValueIter> second) {
      first.Inner().DeepSwap(second.Inner());
    }

  private:
    friend class ProxyIterator<JointProxy<KeyIter, ValueIter> >;

    InnerIterator &Inner() { return inner_; }
    const InnerIterator &Inner() const { return inner_; }
    InnerIterator inner_;
};

template <class Proxy, class Less> class LessWrapper : public std::binary_function<const typename Proxy::value_type &, const typename Proxy::value_type &, bool> {
  public:
    explicit LessWrapper(const Less &less) : less_(less) {}

    bool operator()(const Proxy &left, const Proxy &right) const {
      return less_(left.GetKey(), right.GetKey());
    }
    bool operator()(const Proxy &left, const typename Proxy::value_type &right) const {
      return less_(left.GetKey(), right.GetKey());
    }
    bool operator()(const typename Proxy::value_type &left, const Proxy &right) const {
      return less_(left.GetKey(), right.GetKey());
    }
    bool operator()(const typename Proxy::value_type &left, const typename Proxy::value_type &right) const {
      return less_(left.GetKey(), right.GetKey());
    }

  private:
    const Less less_;
};

} // namespace detail

template <class KeyIter, class ValueIter> class PairedIterator : public ProxyIterator<detail::JointProxy<KeyIter, ValueIter> > {
  public:
    PairedIterator(const KeyIter &key, const ValueIter &value) :
      ProxyIterator<detail::JointProxy<KeyIter, ValueIter> >(detail::JointProxy<KeyIter, ValueIter>(key, value)) {}
};

template <class KeyIter, class ValueIter, class Less> void JointSort(const KeyIter &key_begin, const KeyIter &key_end, const ValueIter &value_begin, const Less &less) {
  ProxyIterator<detail::JointProxy<KeyIter, ValueIter> > full_begin(detail::JointProxy<KeyIter, ValueIter>(key_begin, value_begin));
  detail::LessWrapper<detail::JointProxy<KeyIter, ValueIter>, Less> less_wrap(less);
  std::sort(full_begin, full_begin + (key_end - key_begin), less_wrap);
}


template <class KeyIter, class ValueIter> void JointSort(const KeyIter &key_begin, const KeyIter &key_end, const ValueIter &value_begin) {
  JointSort(key_begin, key_end, value_begin, std::less<typename std::iterator_traits<KeyIter>::value_type>());
}

} // namespace util

#endif // UTIL_JOINT_SORT_H
