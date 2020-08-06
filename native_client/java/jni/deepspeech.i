%module impl

%{
#define SWIG_FILE_WITH_INIT
#include "../../mozilla_voice_stt.h"
%}

%include "typemaps.i"
%include "enums.swg"
%javaconst(1);

%include "arrays_java.i"
// apply to STT_FeedAudioContent and STT_SpeechToText
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
    STT_FreeMetadata(self);
  }
}

%nodefaultctor Metadata;
%nodefaultdtor Metadata;
%nodefaultctor CandidateTranscript;
%nodefaultdtor CandidateTranscript;
%nodefaultctor TokenMetadata;
%nodefaultdtor TokenMetadata;

%typemap(newfree) char* "STT_FreeString($1);";
%newobject STT_SpeechToText;
%newobject STT_IntermediateDecode;
%newobject STT_FinishStream;
%newobject STT_ErrorCodeToErrorMessage;

%rename ("%(strip:[STT_])s") "";

// make struct members camel case to suit Java conventions
%rename ("%(camelcase)s", %$ismember) "";

// ignore automatically generated getTokens and getTranscripts since they don't
// do anything useful and we have already provided getToken(int i) and
// getTranscript(int i) above.
%ignore "Metadata::transcripts";
%ignore "CandidateTranscript::tokens";

%include "../mozilla_voice_stt.h"
