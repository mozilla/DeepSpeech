%module model
%include "typemaps.i"

%{
#define SWIG_FILE_WITH_INIT
#include <string.h>
#include <node_buffer.h>
#include "deepspeech.h"

using namespace v8;
using namespace node;
%}

// convert Node Buffer into a C ptr + length
%typemap(in) (short* IN_ARRAY1, int DIM1)
{
  Local<Object> bufferObj = $input->ToObject();
  char* bufferData = Buffer::Data(bufferObj);
  size_t bufferLength = Buffer::Length(bufferObj);

  $1 = ($1_ltype)bufferData;
  $2 = ($2_ltype)bufferLength;
}

// apply to DS_FeedAudioContent and DS_SpeechToText
%apply (short* IN_ARRAY1, int DIM1) {(const short* aBuffer, unsigned int aBufferSize)};


// convert DS_AudioToInputVector return values to a Node Buffer
%typemap(in,numinputs=0)
  (float** ARGOUTVIEWM_ARRAY2, unsigned int* DIM1, unsigned int* DIM2)
  (float* data_temp, unsigned int dim1_temp, unsigned int dim2_temp)
{
  $1 = &data_temp;
  $2 = &dim1_temp;
  $3 = &dim2_temp;
}
%typemap(argout)
  (float** ARGOUTVIEWM_ARRAY2, unsigned int* DIM1, unsigned int* DIM2)
{
  Handle<Array> array = Array::New(Isolate::GetCurrent(), *$2);
  for (unsigned int i = 0, idx = 0; i < *$2; i++) {
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

%apply (float** ARGOUTVIEWM_ARRAY2, unsigned int* DIM1, unsigned int* DIM2) {(float** aMfcc, unsigned int* aNFrames, unsigned int* aFrameLen)};

// make sure the string returned by SpeechToText is freed
%typemap(newfree) char* "free($1);";
%newobject DS_SpeechToText;
%newobject DS_IntermediateDecode;
%newobject DS_FinishStream;

// convert double pointer retval in CreateModel to an output
%typemap(in, numinputs=0) ModelState **retval (ModelState *ret) {
  ret = NULL;
  $1 = &ret;
}

%typemap(argout) ModelState **retval {
  $result = SWIGV8_ARRAY_NEW();
  SWIGV8_AppendOutput($result, SWIG_From_int(result));
  // owned by SWIG, ModelState destructor gets called when the Python object is finalized (see below)
  %append_output(SWIG_NewPointerObj(%as_voidptr(*$1), $*1_descriptor, SWIG_POINTER_OWN));
}


// convert double pointer retval in SetupStream to an output
%typemap(in, numinputs=0) StreamingState **retval (StreamingState *ret) {
  ret = NULL;
  $1 = &ret;
}

%typemap(argout) StreamingState **retval {
  $result = SWIGV8_ARRAY_NEW();
  SWIGV8_AppendOutput($result, SWIG_From_int(result));
  // not owned, DS_FinishStream deallocates StreamingState
  %append_output(SWIG_NewPointerObj(%as_voidptr(*$1), $*1_descriptor, 0));
}

// extend ModelState with a destructor so that DestroyModel will be called
// when the Python object gets finalized.
%nodefaultctor ModelState;
%nodefaultdtor ModelState;

struct ModelState {};

%extend ModelState {
  ~ModelState() {
    DS_DestroyModel($self);
  }
}

%rename ("%(strip:[DS_])s") "";

%include "../deepspeech.h"
