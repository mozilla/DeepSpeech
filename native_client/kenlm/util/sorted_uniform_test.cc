#include "util/sorted_uniform.hh"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/scoped_array.hpp>
#include <boost/unordered_map.hpp>

#define BOOST_TEST_MODULE SortedUniformTest
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <limits>
#include <vector>

namespace util {
namespace {

template <class KeyT, class ValueT> struct Entry {
  typedef KeyT Key;
  typedef ValueT Value;

  Key key;
  Value value;

  Key GetKey() const {
    return key;
  }

  Value GetValue() const {
    return value;
  }

  bool operator<(const Entry<Key,Value> &other) const {
    return key < other.key;
  }
};

template <class KeyT> struct Accessor {
  typedef KeyT Key;
  template <class Value> Key operator()(const Entry<Key, Value> *entry) const {
    return entry->GetKey();
  }
};

template <class Key, class Value> void Check(const Entry<Key, Value> *begin, const Entry<Key, Value> *end, const boost::unordered_map<Key, Value> &reference, const Key key) {
  typename boost::unordered_map<Key, Value>::const_iterator ref = reference.find(key);
  typedef const Entry<Key, Value> *It;
  // g++ can't tell that require will crash and burn.
  It i = NULL;
  bool ret = SortedUniformFind<It, Accessor<Key>, Pivot64>(Accessor<Key>(), begin, end, key, i);
  if (ref == reference.end()) {
    BOOST_CHECK(!ret);
  } else {
    BOOST_REQUIRE(ret);
    BOOST_CHECK_EQUAL(ref->second, i->GetValue());
  }
}

BOOST_AUTO_TEST_CASE(empty) {
  typedef const Entry<uint64_t, float> T;
  const T *i;
  bool ret = SortedUniformFind<const T*, Accessor<uint64_t>, Pivot64>(Accessor<uint64_t>(), (const T*)NULL, (const T*)NULL, (uint64_t)10, i);
  BOOST_CHECK(!ret);
}

template <class Key> void RandomTest(Key upper, size_t entries, size_t queries) {
  typedef unsigned char Value;
  boost::mt19937 rng;
  boost::uniform_int<Key> range_key(0, upper);
  boost::uniform_int<Value> range_value(0, 255);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<Key> > gen_key(rng, range_key);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<unsigned char> > gen_value(rng, range_value);

  typedef Entry<Key, Value> Ent;
  std::vector<Ent> backing;
  boost::unordered_map<Key, unsigned char> reference;
  Ent ent;
  for (size_t i = 0; i < entries; ++i) {
    Key key = gen_key();
    unsigned char value = gen_value();
    if (reference.insert(std::make_pair(key, value)).second) {
      ent.key = key;
      ent.value = value;
      backing.push_back(ent);
    }
  }
  std::sort(backing.begin(), backing.end());

  // Random queries.
  for (size_t i = 0; i < queries; ++i) {
    const Key key = gen_key();
    Check<Key, unsigned char>(&*backing.begin(), &*backing.end(), reference, key);
  }

  typename boost::unordered_map<Key, unsigned char>::const_iterator it = reference.begin();
  for (size_t i = 0; (i < queries) && (it != reference.end()); ++i, ++it) {
    Check<Key, unsigned char>(&*backing.begin(), &*backing.end(), reference, it->second);
  }
}

BOOST_AUTO_TEST_CASE(basic) {
  RandomTest<uint8_t>(11, 10, 200);
}

BOOST_AUTO_TEST_CASE(tiny_dense_random) {
  RandomTest<uint8_t>(11, 50, 200);
}

BOOST_AUTO_TEST_CASE(small_dense_random) {
  RandomTest<uint8_t>(100, 100, 200);
}

BOOST_AUTO_TEST_CASE(small_sparse_random) {
  RandomTest<uint8_t>(200, 15, 200);
}

BOOST_AUTO_TEST_CASE(medium_sparse_random) {
  RandomTest<uint16_t>(32000, 1000, 2000);
}

BOOST_AUTO_TEST_CASE(sparse_random) {
  RandomTest<uint64_t>(std::numeric_limits<uint64_t>::max(), 100000, 2000);
}

} // namespace
} // namespace util
