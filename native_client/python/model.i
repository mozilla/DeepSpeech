%module(threads="1") model

%{
#define SWIG_FILE_WITH_INIT
#include "deepspeech.h"
%}

%include "numpy.i"
%init %{
import_array();
%}

%apply (short* IN_ARRAY1, int DIM1) {(const short* aBuffer, unsigned int aBufferSize)};
%apply (float** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {(float** aMfcc, int* aNFrames, int* aFrameLen)};
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* aMfcc, int aNFrames, int aFrameLen)};

%include "../deepspeech.h"
