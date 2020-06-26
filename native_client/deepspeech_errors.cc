#include "deepspeech.h"
#include <string.h>

char*
DS_ErrorCodeToErrorMessage(int aErrorCode)
{
#define RETURN_MESSAGE(NAME, VALUE, DESC) \
    case NAME: \
      return strdup(DESC);

  switch(aErrorCode)
  {
    DS_FOR_EACH_ERROR(RETURN_MESSAGE)
    default:
      return strdup("Unknown error, please make sure you are using the correct native binary.");
  }

#undef RETURN_MESSAGE
}
