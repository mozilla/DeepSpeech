%module impl

%{
#define SWIG_FILE_WITH_INIT
#include "../../deepspeech.h"
%}

%include "typemaps.i"
%include "enums.swg"
%javaconst(1);

%include "arrays_java.i"
// apply to DS_FeedAudioContent and DS_SpeechToText
%apply short[] { short* };

%include "cpointer.i"
%pointer_functions(ModelState*, modelstatep);
%pointer_functions(StreamingState*, streamingstatep);

%extend struct CandidateTranscript {
  /**
   * Retrieve one TokenMetadata element
   * 
   * @param i Array index of the TokenMetadata to get
   *
   * @return The TokenMetadata requested or null
   */
  const TokenMetadata& getToken(int i) {
    return self->tokens[i];
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
  const CandidateTranscript& getTranscript(int i) {
    return self->transcripts[i];
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

%typemap(newfree) char* "DS_FreeString($1);";
%newobject DS_SpeechToText;
%newobject DS_IntermediateDecode;
%newobject DS_FinishStream;
%newobject DS_ErrorCodeToErrorMessage;

%rename ("%(strip:[DS_])s") "";

// make struct members camel case to suit Java conventions
%rename ("%(camelcase)s", %$ismember) "";

// ignore automatically generated getTokens and getTranscripts since they don't
// do anything useful and we have already provided getToken(int i) and
// getTranscript(int i) above.
%ignore "Metadata::transcripts";
%ignore "CandidateTranscript::tokens";

%include "../deepspeech.h"
