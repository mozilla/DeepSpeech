#ifndef LM_BUILDER_NGRAM_STREAM_H
#define LM_BUILDER_NGRAM_STREAM_H

#include "lm/common/ngram.hh"
#include "util/stream/chain.hh"
#include "util/stream/multi_stream.hh"
#include "util/stream/stream.hh"

#include <cstddef>

namespace lm {

template <class Proxy> class ProxyStream {
  public:
    // Make an invalid stream.
    ProxyStream() {}

    explicit ProxyStream(const util::stream::ChainPosition &position, const Proxy &proxy = Proxy())
      : proxy_(proxy), stream_(position) {
      proxy_.ReBase(stream_.Get());
    }

    Proxy &operator*() { return proxy_; }
    const Proxy &operator*() const { return proxy_; }

    Proxy *operator->() { return &proxy_; }
    const Proxy *operator->() const { return &proxy_; }

    void *Get() { return stream_.Get(); }
    const void *Get() const { return stream_.Get(); }

    operator bool() const { return stream_; }
    bool operator!() const { return !stream_; }
    void Poison() { stream_.Poison(); }

    ProxyStream<Proxy> &operator++() {
      ++stream_;
      proxy_.ReBase(stream_.Get());
      return *this;
    }

  private:
    Proxy proxy_;
    util::stream::Stream stream_;
};

template <class Payload> class NGramStream : public ProxyStream<NGram<Payload> > {
  public:
    // Make an invalid stream.
    NGramStream() {}

    explicit NGramStream(const util::stream::ChainPosition &position) :
      ProxyStream<NGram<Payload> >(position, NGram<Payload>(NULL, NGram<Payload>::OrderFromSize(position.GetChain().EntrySize()))) {}
};

template <class Payload> class NGramStreams : public util::stream::GenericStreams<NGramStream<Payload> > {
  private:
    typedef util::stream::GenericStreams<NGramStream<Payload> > P;
  public:
    NGramStreams() : P() {}
    NGramStreams(const util::stream::ChainPositions &positions) : P(positions) {}
};

} // namespace
#endif // LM_BUILDER_NGRAM_STREAM_H
