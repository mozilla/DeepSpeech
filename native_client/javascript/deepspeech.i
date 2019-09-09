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
  Local<Object> bufferObj = SWIGV8_TO_OBJECT($input);
  char* bufferData = Buffer::Data(bufferObj);
  size_t bufferLength = Buffer::Length(bufferObj);

  $1 = ($1_ltype)bufferData;
  $2 = ($2_ltype)bufferLength;
}

// apply to DS_FeedAudioContent and DS_SpeechToText
%apply (short* IN_ARRAY1, int DIM1) {(const short* aBuffer, unsigned int aBufferSize)};


// make sure the string returned by SpeechToText is freed
%typemap(newfree) char* "DS_FreeString($1);";

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
  // owned by SWIG, ModelState destructor gets called when the JavaScript object is finalized (see below)
  %append_output(SWIG_NewPointerObj(%as_voidptr(*$1), $*1_descriptor, SWIG_POINTER_OWN));
}


// convert double pointer retval in CreateStream to an output
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

%nodefaultctor ModelState;
%nodefaultdtor ModelState;

%typemap(out) MetadataItem* %{
  $result = SWIGV8_ARRAY_NEW();
  for (int i = 0; i < arg1->num_items; ++i) {
    SWIGV8_AppendOutput($result, SWIG_NewPointerObj(SWIG_as_voidptr(&result[i]), SWIGTYPE_p_MetadataItem, SWIG_POINTER_OWN));
  }
%}

%nodefaultdtor Metadata;
%nodefaultctor Metadata;
%nodefaultctor MetadataItem;
%nodefaultdtor MetadataItem;

%extend struct Metadata {
  ~Metadata() {
    DS_FreeMetadata($self);
  }
}

%extend struct MetadataItem {
  ~MetadataItem() { }
}

%rename ("%(strip:[DS_])s") "";

%include "../deepspeech.h"
