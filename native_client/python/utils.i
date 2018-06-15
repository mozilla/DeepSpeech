%module utils

%{
#define SWIG_FILE_WITH_INIT
#include "deepspeech_utils.h"
%}

%include "numpy.i"
%init %{
import_array();
%}

%apply (short* IN_ARRAY1, int DIM1) {(const short* aBuffer, unsigned int aBufferSize)};
%apply (float** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {(float** aMfcc, int* aNFrames, int* aFrameLen)};

%include "../deepspeech_utils.h"
