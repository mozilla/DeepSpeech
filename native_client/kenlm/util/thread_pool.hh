#ifndef UTIL_THREAD_POOL_H
#define UTIL_THREAD_POOL_H

#include "util/pcqueue.hh"

#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/optional.hpp>
#include <boost/thread.hpp>

#include <iostream>
#include <cstdlib>

namespace util {

template <class HandlerT> class Worker : boost::noncopyable {
  public:
    typedef HandlerT Handler;
    typedef typename Handler::Request Request;

    template <class Construct> Worker(PCQueue<Request> &in, Construct &construct, const Request &poison)
      : in_(in), handler_(construct), poison_(poison), thread_(boost::ref(*this)) {}

    // Only call from thread.
    void operator()() {
      Request request;
      while (1) {
        in_.Consume(request);
        if (request == poison_) return;
        try {
          (*handler_)(request);
        }
        catch(const std::exception &e) {
          std::cerr << "Handler threw " << e.what() << std::endl;
          abort();
        }
        catch(...) {
          std::cerr << "Handler threw an exception, dropping request" << std::endl;
          abort();
        }
      }
    }

    void Join() {
      thread_.join();
    }

  private:
    PCQueue<Request> &in_;

    boost::optional<Handler> handler_;

    const Request poison_;

    boost::thread thread_;
};

template <class HandlerT> class ThreadPool : boost::noncopyable {
  public:
    typedef HandlerT Handler;
    typedef typename Handler::Request Request;

    template <class Construct> ThreadPool(std::size_t queue_length, std::size_t workers, Construct handler_construct, Request poison) : in_(queue_length), poison_(poison) {
      for (size_t i = 0; i < workers; ++i) {
        workers_.push_back(new Worker<Handler>(in_, handler_construct, poison));
      }
    }

    ~ThreadPool() {
      for (std::size_t i = 0; i < workers_.size(); ++i) {
        Produce(poison_);
      }
      for (typename boost::ptr_vector<Worker<Handler> >::iterator i = workers_.begin(); i != workers_.end(); ++i) {
        i->Join();
      }
    }

    void Produce(const Request &request) {
      in_.Produce(request);
    }

    // For adding to the queue.
    PCQueue<Request> &In() { return in_; }

  private:
    PCQueue<Request> in_;

    boost::ptr_vector<Worker<Handler> > workers_;

    Request poison_;
};

template <class Handler> class RecyclingHandler {
  public:
    typedef typename Handler::Request Request;

    template <class Construct> RecyclingHandler(PCQueue<Request> &recycling, Construct &handler_construct)
      : inner_(handler_construct), recycling_(recycling) {}

    void operator()(Request &request) {
      inner_(request);
      recycling_.Produce(request);
    }

  private:
    Handler inner_;
    PCQueue<Request> &recycling_;
};

template <class HandlerT> class RecyclingThreadPool : boost::noncopyable {
  public:
    typedef HandlerT Handler;
    typedef typename Handler::Request Request;

    // Remember to call PopulateRecycling afterwards in most cases.
    template <class Construct> RecyclingThreadPool(std::size_t queue, std::size_t workers, Construct handler_construct, Request poison)
      : recycling_(queue), pool_(queue, workers, RecyclingHandler<Handler>(recycling_, handler_construct), poison) {}

    // Initialization: put stuff into the recycling queue.  This could also be
    // done by calling Produce without Consume, but it's often easier to
    // initialize with PopulateRecycling then do a Consume/Produce loop.
    void PopulateRecycling(const Request &request) {
      recycling_.Produce(request);
    }

    Request Consume() {
      return recycling_.Consume();
    }

    void Produce(const Request &request) {
      pool_.Produce(request);
    }
    
  private:
    PCQueue<Request> recycling_;
    ThreadPool<RecyclingHandler<Handler> > pool_;
};

} // namespace util

#endif // UTIL_THREAD_POOL_H
