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
%array_functions(struct MetadataItem, metadataItem_array);

%extend struct Metadata {
  MetadataItem getItem(int i) {
    return metadataItem_array_getitem(self->items, i);
  }

  ~Metadata() {
    DS_FreeMetadata(self);
  }
}

%nodefaultdtor Metadata;
%nodefaultctor Metadata;
%nodefaultctor MetadataItem;
%nodefaultdtor MetadataItem;

%newobject DS_SpeechToText;
%newobject DS_IntermediateDecode;
%newobject DS_FinishStream;

%rename ("%(strip:[DS_])s") "";

%include "../deepspeech.h"
