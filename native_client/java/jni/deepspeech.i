%module impl

%{
#define SWIG_FILE_WITH_INIT
#include "../../deepspeech.h"
%}

%include "typemaps.i"

%include "arrays_java.i"
// apply to DS_FeedAudioContent and DS_SpeechToText
%apply short[] { short* };

%include "cpointer.i"
%pointer_functions(ModelState*, modelstatep);
%pointer_functions(StreamingState*, streamingstatep);

%typemap(newfree) char* "DS_FreeString($1);";

%include "carrays.i"
%array_functions(struct TokenMetadata, TokenMetadata_array);
%array_functions(struct CandidateTranscript, CandidateTranscript_array);

%extend struct CandidateTranscript {
  /**
   * Retrieve one TokenMetadata element
   * 
   * @param i Array index of the TokenMetadata to get
   *
   * @return The TokenMetadata requested or null
   */
  TokenMetadata getToken(int i) {
    return TokenMetadata_array_getitem(self->tokens, i);
  }
}

%extend struct Metadata {
  /**
   * Retrieve one CandidateTranscript element
   * 
   * @param i Array index of the CandidateTranscript to get
   *
   * @return The CandidateTranscript requested or null
   */
  CandidateTranscript getTranscript(int i) {
    return CandidateTranscript_array_getitem(self->transcripts, i);
  }

  ~Metadata() {
    DS_FreeMetadata(self);
  }
}

%nodefaultctor Metadata;
%nodefaultdtor Metadata;
%nodefaultctor CandidateTranscript;
%nodefaultdtor CandidateTranscript;
%nodefaultctor TokenMetadata;
%nodefaultdtor TokenMetadata;

%newobject DS_SpeechToText;
%newobject DS_IntermediateDecode;
%newobject DS_FinishStream;

%rename ("%(strip:[DS_])s") "";

%include "../deepspeech.h"
