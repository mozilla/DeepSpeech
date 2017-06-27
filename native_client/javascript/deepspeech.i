%module model
%include "typemaps.i"

%{
#define SWIG_FILE_WITH_INIT
#include <string.h>
#include <node_buffer.h>
#include "deepspeech.h"
#include "deepspeech_utils.h"

using namespace v8;
using namespace node;
%}

%typemap(in) (short* IN_ARRAY1, int DIM1)
{
  Local<Object> bufferObj = $input->ToObject();
  char* bufferData = Buffer::Data(bufferObj);
  size_t bufferLength = Buffer::Length(bufferObj);

  $1 = ($1_ltype)bufferData;
  $2 = ($2_ltype)bufferLength;
}

%typemap(in) (float* IN_ARRAY2, int DIM1, int DIM2) (float* mfcc = NULL)
{
  Local<Array> array = Local<Array>::Cast($input);

  $2 = array->Length();
  $3 = 0;

  for (int i = 0, idx = 0; i < $2; i++) {
    Local<Float32Array> dataObj = Local<Float32Array>::Cast(array->Get(i));
    if (i == 0) {
      $3 = dataObj->Length();
      mfcc = (float*)malloc(sizeof(float) * $2 * $3);
    }
    memcpy(mfcc + (idx += $3), dataObj->Buffer()->GetContents().Data(), $3);
  }

  $1 = mfcc;
}
%typemap(freearg) (float* IN_ARRAY2, int DIM1, int DIM2)
{
  if (mfcc$argnum) {
    free(mfcc$argnum);
  }
}

%typemap(in,numinputs=0)
  (float** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2)
  (float* data_temp, int dim1_temp, int dim2_temp)
{
  $1 = &data_temp;
  $2 = &dim1_temp;
  $3 = &dim2_temp;
}
%typemap(argout)
  (float** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2)
{
  Handle<Array> array = Array::New(Isolate::GetCurrent(), *$2);
  for (int i = 0, idx = 0; i < *$2; i++) {
    Handle<ArrayBuffer> buffer =
      ArrayBuffer::New(Isolate::GetCurrent(), *$1, *$3 * sizeof(float));
    memcpy(buffer->GetContents().Data(),
           (*$1) + (idx += *$3), *$3 * sizeof(float));
    Handle<Float32Array> inner = Float32Array::New(buffer, 0, *$3);
    array->Set(i, inner);
  }
  free(*$1);
  $result = array;
}

%apply (short* IN_ARRAY1, int DIM1) {(const short* aBuffer, unsigned int aBufferSize)};
%apply (float** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {(float** aMfcc, int* aNFrames, int* aFrameLen)};
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* aMfcc, int aNFrames, int aFrameLen)};

%include "../deepspeech.h"
%include "../deepspeech_utils.h"
