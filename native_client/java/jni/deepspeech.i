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

%rename ("%(strip:[DS_])s") "";

%include "../deepspeech.h"
