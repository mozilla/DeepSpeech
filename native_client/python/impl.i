%module(threads="1") impl

%{
#define SWIG_FILE_WITH_INIT
#include "deepspeech.h"
%}

%include "numpy.i"
%init %{
import_array();
%}

// apply NumPy conversion typemap to DS_FeedAudioContent and DS_SpeechToText
%apply (short* IN_ARRAY1, int DIM1) {(const short* aBuffer, unsigned int aBufferSize)};

%typemap(in, numinputs=0) ModelState **retval (ModelState *ret) {
  ret = NULL;
  $1 = &ret;
}

%typemap(argout) ModelState **retval {
  // not owned, Python wrapper in __init__.py calls DS_FreeModel
  %append_output(SWIG_NewPointerObj(%as_voidptr(*$1), $*1_descriptor, 0));
}

%typemap(in, numinputs=0) StreamingState **retval (StreamingState *ret) {
  ret = NULL;
  $1 = &ret;
}

%typemap(argout) StreamingState **retval {
  // not owned, DS_FinishStream deallocates StreamingState
  %append_output(SWIG_NewPointerObj(%as_voidptr(*$1), $*1_descriptor, 0));
}

%typemap(out) Metadata* {
  // owned, extended destructor needs to be called by SWIG
  %append_output(SWIG_NewPointerObj(%as_voidptr($1), $1_descriptor, SWIG_POINTER_OWN));
}

%fragment("parent_reference_init", "init") {
  // Thread-safe initialization - initialize during Python module initialization
  parent_reference();
}

%fragment("parent_reference_function", "header", fragment="parent_reference_init") {

static PyObject *parent_reference() {
  static PyObject *parent_reference_string = SWIG_Python_str_FromChar("__parent_reference");
  return parent_reference_string;
}

}

%typemap(out, fragment="parent_reference_function") CandidateTranscript* %{
  $result = PyList_New(arg1->num_transcripts);
  for (int i = 0; i < arg1->num_transcripts; ++i) {
    PyObject* o = SWIG_NewPointerObj(SWIG_as_voidptr(&arg1->transcripts[i]), SWIGTYPE_p_CandidateTranscript, 0);
    // Add a reference to Metadata in the returned elements to avoid premature
    // garbage collection
    PyObject_SetAttr(o, parent_reference(), $self);
    PyList_SetItem($result, i, o);
  }
%}

%typemap(out, fragment="parent_reference_function") TokenMetadata* %{
  $result = PyList_New(arg1->num_tokens);
  for (int i = 0; i < arg1->num_tokens; ++i) {
    PyObject* o = SWIG_NewPointerObj(SWIG_as_voidptr(&arg1->tokens[i]), SWIGTYPE_p_TokenMetadata, 0);
    // Add a reference to CandidateTranscript in the returned elements to avoid premature
    // garbage collection
    PyObject_SetAttr(o, parent_reference(), $self);
    PyList_SetItem($result, i, o);
  }
%}

%extend struct TokenMetadata {
%pythoncode %{
  def __repr__(self):
    return 'TokenMetadata(text=\'{}\', timestep={}, start_time={})'.format(self.text, self.timestep, self.start_time)
%}
}

%extend struct CandidateTranscript {
%pythoncode %{
  def __repr__(self):
    tokens_repr = ',\n'.join(repr(i) for i in self.tokens)
    tokens_repr = '\n'.join('  ' + l for l in tokens_repr.split('\n'))
    return 'CandidateTranscript(confidence={}, tokens=[\n{}\n])'.format(self.confidence, tokens_repr)
%}
}

%extend struct Metadata {
%pythoncode %{
  def __repr__(self):
    transcripts_repr = ',\n'.join(repr(i) for i in self.transcripts)
    transcripts_repr = '\n'.join('  ' + l for l in transcripts_repr.split('\n'))
    return 'Metadata(transcripts=[\n{}\n])'.format(transcripts_repr)
%}
}

%ignore Metadata::num_transcripts;
%ignore CandidateTranscript::num_tokens;

%extend struct Metadata {
  ~Metadata() {
    DS_FreeMetadata($self);
  }
}

%nodefaultctor Metadata;
%nodefaultdtor Metadata;
%nodefaultctor CandidateTranscript;
%nodefaultdtor CandidateTranscript;
%nodefaultctor TokenMetadata;
%nodefaultdtor TokenMetadata;

%typemap(newfree) char* "DS_FreeString($1);";

%newobject DS_SpeechToText;
%newobject DS_IntermediateDecode;
%newobject DS_FinishStream;
%newobject DS_Version;
%newobject DS_ErrorCodeToErrorMessage;

%rename ("%(strip:[DS_])s") "";

%include "../deepspeech.h"
