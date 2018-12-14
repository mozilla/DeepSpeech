%module swigwrapper

%{
#include "ctc_beam_search_decoder.h"
#define SWIG_FILE_WITH_INIT
%}

%include "pyabc.i"
%include "std_string.i"
%include "std_vector.i"
%include "numpy.i"

%init %{
import_array();
%}

// Convert NumPy arrays to pointer+lengths
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(const double *probs, int time_dim, int class_dim)};
%apply (double* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(const double *probs, int batch_dim, int time_dim, int class_dim)};
%apply (int* IN_ARRAY1, int DIM1) {(const int *seq_lengths, int seq_lengths_size)};

// Convert char* to Alphabet
%rename (ctc_beam_search_decoder) mod_decoder;
%inline %{
std::vector<Output>
mod_decoder(const double *probs,
            int time_dim,
            int class_dim,
            char* alphabet_config_path,
            size_t beam_size,
            double cutoff_prob,
            size_t cutoff_top_n,
            Scorer *ext_scorer)
{
    Alphabet a(alphabet_config_path);
    return ctc_beam_search_decoder(probs, time_dim, class_dim, a, beam_size,
                                   cutoff_prob, cutoff_top_n, ext_scorer);
}
%}

%rename (ctc_beam_search_decoder_batch) mod_decoder_batch;
%inline %{
std::vector<std::vector<Output>>
mod_decoder_batch(const double *probs,
                  int batch_dim,
                  int time_dim,
                  int class_dim,
                  const int *seq_lengths,
                  int seq_lengths_size,
                  char* alphabet_config_path,
                  size_t beam_size,
                  size_t num_processes,
                  double cutoff_prob,
                  size_t cutoff_top_n,
                  Scorer *ext_scorer)
{
    Alphabet a(alphabet_config_path);
    return ctc_beam_search_decoder_batch(probs, batch_dim, time_dim, class_dim,
                                         seq_lengths, seq_lengths_size, a, beam_size,
                                         num_processes, cutoff_prob, cutoff_top_n,
                                         ext_scorer);
}
%}


%ignore Scorer::dictionary;

%include "output.h"
%include "scorer.h"
%include "ctc_beam_search_decoder.h"

%template(IntVector) std::vector<int>;
%template(OutputVector) std::vector<Output>;
%template(OutputVectorVector) std::vector<std::vector<Output>>;
