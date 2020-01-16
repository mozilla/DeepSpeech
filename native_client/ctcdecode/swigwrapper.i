%module swigwrapper

%{
#include "ctc_beam_search_decoder.h"
#define SWIG_FILE_WITH_INIT
#define SWIG_PYTHON_STRICT_BYTE_CHAR
%}

%include "pyabc.i"
%include "std_string.i"
%include "std_vector.i"
%include "numpy.i"

%init %{
import_array();
%}

namespace std {
    %template(StringVector) vector<string>;
}

// Convert NumPy arrays to pointer+lengths
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(const double *probs, int time_dim, int class_dim)};
%apply (double* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(const double *probs, int batch_size, int time_dim, int class_dim)};
%apply (int* IN_ARRAY1, int DIM1) {(const int *seq_lengths, int seq_lengths_size)};

%ignore Scorer::dictionary;

%include "../alphabet.h"
%include "output.h"
%include "scorer.h"
%include "ctc_beam_search_decoder.h"

%template(IntVector) std::vector<int>;
%template(OutputVector) std::vector<Output>;
%template(OutputVectorVector) std::vector<std::vector<Output>>;
