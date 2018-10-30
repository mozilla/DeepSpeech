# See www.openfst.org for extensive documentation on this weighted
# finite-state transducer library.


from libcpp.string cimport string
from basictypes cimport int8
from basictypes cimport int16
from basictypes cimport int32
from basictypes cimport int64
from basictypes cimport uint8
from basictypes cimport uint16
from basictypes cimport uint32
from basictypes cimport uint64


cdef extern from "<iostream>" namespace "std" nogil:

  cdef cppclass iostream:

    pass

  cdef cppclass istream(iostream):

    pass

  cdef cppclass ostream(iostream):

    pass


# We are ignoring openmodes for the moment.


cdef extern from "<fstream>" namespace "std" nogil:

  cdef cppclass ifstream(istream):

    ifstream(const string &)

  cdef cppclass ofstream(ostream):

    ofstream(const string &)


cdef extern from "<sstream>" namespace "std" nogil:

  cdef cppclass stringstream(istream, ostream):

    stringstream()

    string str()

    stringstream &operator<<(const string &)

    stringstream &operator<<(bool)

    # We define these in terms of the Google basictypes.

    stringstream &operator<<(int8)

    stringstream &operator<<(uint8)

    stringstream &operator<<(int16)

    stringstream &operator<<(uint16)

    stringstream &operator<<(int32)

    stringstream &operator<<(uint32)

    stringstream &operator<<(int64)

    stringstream &operator<<(uint64)

    stringstream &operator<<(double)

    stringstream &operator<<(long double)
