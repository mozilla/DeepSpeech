# See www.openfst.org for extensive documentation on this weighted
# finite-state transducer library.


from libc.stdint cimport *


cdef extern from "<fst/types.h>" nogil:

  ctypedef int8_t int8
  ctypedef int16_t int16
  ctypedef int32_t int32
  ctypedef int64_t int64
  ctypedef uint8_t uint8
  ctypedef uint16_t uint16
  ctypedef uint32_t uint32
  ctypedef uint64_t uint64
