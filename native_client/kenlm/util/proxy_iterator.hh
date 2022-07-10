#ifndef UTIL_PROXY_ITERATOR_H
#define UTIL_PROXY_ITERATOR_H

#include <cstddef>
#include <iterator>

/* This is a RandomAccessIterator that uses a proxy to access the underlying
 * data.  Useful for packing data at bit offsets but still using STL
 * algorithms.
 *
 * Normally I would use boost::iterator_facade but some people are too lazy to
 * install boost and still want to use my language model.  It's amazing how
 * many operators an iterator has.
 *
 * The Proxy needs to provide:
 *   class InnerIterator;
 *   InnerIterator &Inner();
 *   const InnerIterator &Inner() const;
 *
 * InnerIterator has to implement:
 *   operator==(InnerIterator)
 *   operator<(InnerIterator)
 *   operator+=(std::ptrdiff_t)
 *   operator-(InnerIterator)
 * and of course whatever Proxy needs to dereference it.
 *
 * It's also a good idea to specialize std::swap for Proxy.
 */

namespace util {
template <class Proxy> class ProxyIterator {
  private:
    // Self.
    typedef ProxyIterator<Proxy> S;
    typedef typename Proxy::InnerIterator InnerIterator;

  public:
    typedef std::random_access_iterator_tag iterator_category;
    typedef typename Proxy::value_type value_type;
    typedef std::ptrdiff_t difference_type;
    typedef Proxy reference;
    typedef ProxyIterator<Proxy> * pointer;

    ProxyIterator() {}

    // For cast from non const to const.
    template <class AlternateProxy> ProxyIterator(const ProxyIterator<AlternateProxy> &in) : p_(*in) {}
    explicit ProxyIterator(const Proxy &p) : p_(p) {}

/*    // p_'s swap does value swapping, but here we want iterator swapping
    friend inline void swap(ProxyIterator<Proxy> &first, ProxyIterator<Proxy> &second) {
      swap(first.I(), second.I());
    }*/

    // p_'s operator= does value copying, but here we want iterator copying.
    S &operator=(const S &other) {
      I() = other.I();
      return *this;
    }

    bool operator==(const S &other) const { return I() == other.I(); }
    bool operator!=(const S &other) const { return !(*this == other); }
    bool operator<(const S &other) const { return I() < other.I(); }
    bool operator>(const S &other) const { return other < *this; }
    bool operator<=(const S &other) const { return !(*this > other); }
    bool operator>=(const S &other) const { return !(*this < other); }

    S &operator++() { return *this += 1; }
    S operator++(int) { S ret(*this); ++*this; return ret; }
    S &operator+=(std::ptrdiff_t amount) { I() += amount; return *this; }
    S operator+(std::ptrdiff_t amount) const { S ret(*this); ret += amount; return ret; }

    S &operator--() { return *this -= 1; }
    S operator--(int) { S ret(*this); --*this; return ret; }
    S &operator-=(std::ptrdiff_t amount) { I() += (-amount); return *this; }
    S operator-(std::ptrdiff_t amount) const { S ret(*this); ret -= amount; return ret; }

    std::ptrdiff_t operator-(const S &other) const { return I() - other.I(); }

    Proxy operator*() const { return p_; }
    Proxy *operator->() { return &p_; }
    const Proxy *operator->() const { return &p_; }
    Proxy operator[](std::ptrdiff_t amount) const { return *(*this + amount); }

    const InnerIterator &Inner() { return p_.Inner(); }

  private:
    InnerIterator &I() { return p_.Inner(); }
    const InnerIterator &I() const { return p_.Inner(); }

    Proxy p_;
};

template <class Proxy> ProxyIterator<Proxy> operator+(std::ptrdiff_t amount, const ProxyIterator<Proxy> &it) {
  return it + amount;
}

} // namespace util

#endif // UTIL_PROXY_ITERATOR_H
