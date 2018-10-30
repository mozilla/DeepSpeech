# See www.openfst.org for extensive documentation on this weighted
# finite-state transducer library.


from libcpp.memory cimport shared_ptr


# This is mysteriously missing from libcpp.memory.

cdef extern from "<memory>" namespace "std" nogil:

  shared_ptr[T] static_pointer_cast[T, U](const shared_ptr[U] &)
