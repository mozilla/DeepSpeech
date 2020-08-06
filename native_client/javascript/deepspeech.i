%module model
%include "typemaps.i"

%{
#define SWIG_FILE_WITH_INIT
#include <string.h>
#include <node_buffer.h>
#include "mozilla_voice_stt.h"

using namespace v8;
using namespace node;
%}

// convert Node Buffer into a C ptr + length
%typemap(in) (short* IN_ARRAY1, int DIM1)
{
  Local<Object> bufferObj = SWIGV8_TO_OBJECT($input);
  char* bufferData = Buffer::Data(bufferObj);
  size_t bufferLength = Buffer::Length(bufferObj);

  if (bufferLength % 2 != 0) {
    SWIG_exception_fail(SWIG_ERROR, "Buffer length must be even. Make sure your input audio is 16-bits per sample.");
  }

  $1 = ($1_ltype)bufferData;
  $2 = ($2_ltype)(bufferLength / 2);
}

// apply to STT_FeedAudioContent and STT_SpeechToText
%apply (short* IN_ARRAY1, int DIM1) {(const short* aBuffer, unsigned int aBufferSize)};


// make sure the string returned by SpeechToText is freed
%typemap(newfree) char* "STT_FreeString($1);";

%newobject STT_SpeechToText;
%newobject STT_IntermediateDecode;
%newobject STT_FinishStream;
%newobject STT_Version;
%newobject STT_ErrorCodeToErrorMessage;

// convert double pointer retval in CreateModel to an output
%typemap(in, numinputs=0) ModelState **retval (ModelState *ret) {
  ret = NULL;
  $1 = &ret;
}

%typemap(argout) ModelState **retval {
  $result = SWIGV8_ARRAY_NEW();
  SWIGV8_AppendOutput($result, SWIG_From_int(result));
  // owned by the application. NodeJS does not guarantee the finalizer will be called so applications must call FreeMetadata themselves.
  %append_output(SWIG_NewPointerObj(%as_voidptr(*$1), $*1_descriptor, 0));
}


// convert double pointer retval in CreateStream to an output
%typemap(in, numinputs=0) StreamingState **retval (StreamingState *ret) {
  ret = NULL;
  $1 = &ret;
}

%typemap(argout) StreamingState **retval {
  $result = SWIGV8_ARRAY_NEW();
  SWIGV8_AppendOutput($result, SWIG_From_int(result));
  // not owned, STT_FinishStream deallocates StreamingState
  %append_output(SWIG_NewPointerObj(%as_voidptr(*$1), $*1_descriptor, 0));
}

%nodefaultctor ModelState;
%nodefaultdtor ModelState;

%typemap(out) TokenMetadata* %{
  $result = SWIGV8_ARRAY_NEW();
  for (int i = 0; i < arg1->num_tokens; ++i) {
    SWIGV8_AppendOutput($result, SWIG_NewPointerObj(SWIG_as_voidptr(&result[i]), SWIGTYPE_p_TokenMetadata, 0));
  }
%}

%typemap(out) CandidateTranscript* %{
  $result = SWIGV8_ARRAY_NEW();
  for (int i = 0; i < arg1->num_transcripts; ++i) {
    SWIGV8_AppendOutput($result, SWIG_NewPointerObj(SWIG_as_voidptr(&result[i]), SWIGTYPE_p_CandidateTranscript, 0));
  }
%}

%ignore Metadata::num_transcripts;
%ignore CandidateTranscript::num_tokens;

%nodefaultctor Metadata;
%nodefaultdtor Metadata;
%nodefaultctor CandidateTranscript;
%nodefaultdtor CandidateTranscript;
%nodefaultctor TokenMetadata;
%nodefaultdtor TokenMetadata;

%rename ("%(strip:[STT_])s") "";

%include "../mozilla_voice_stt.h"
