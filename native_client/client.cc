#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <sox.h>
#include "deepspeech.h"

#define N_CEP 26
#define N_CONTEXT 9

int
main(int argc, char **argv)
{
  if (argc < 3) {
    printf("Usage: deepspeech [model path] [audio path]\n");
    return 1;
  }

  // Initialise DeepSpeech
  DeepSpeechContext* ctx = DsInit(argv[1], N_CEP, N_CONTEXT);
  assert(ctx);

  // Initialise SOX
  assert(sox_init() == SOX_SUCCESS);

  sox_format_t* input = sox_open_read(argv[2], NULL, NULL, NULL);
  assert(input);

  int sampleRate = (int)input->signal.rate;

  // Resample/reformat the audio so we can pass it through the MFCC functions
  sox_signalinfo_t target_signal = {
      SOX_UNSPEC, // Rate
      1, // Channels
      16, // Precision
      SOX_UNSPEC, // Length
      NULL // Effects headroom multiplier
  };

  sox_encodinginfo_t target_encoding = {
    SOX_ENCODING_SIGN2, // Sample format
    16, // Bits per sample
    0.0, // Compression factor
    sox_option_default, // Should bytes be reversed
    sox_option_default, // Should nibbles be reversed
    sox_option_default, // Should bits be reversed (pairs of bits?)
    sox_false // Reverse endianness
  };

  char *buffer;
  size_t buffer_size;
  sox_format_t* output = sox_open_memstream_write(&buffer, &buffer_size,
                                                  &target_signal,
                                                  &target_encoding,
                                                  "raw", NULL);
  assert(output);

  // Setup the effects chain to decode/resample
  char* sox_args[10];
  sox_effects_chain_t* chain =
    sox_create_effects_chain(&input->encoding, &output->encoding);

  sox_effect_t* e = sox_create_effect(sox_find_effect("input"));
  sox_args[0] = (char*)input;
  assert(sox_effect_options(e, 1, sox_args) == SOX_SUCCESS);
  assert(sox_add_effect(chain, e, &input->signal, &input->signal) ==
         SOX_SUCCESS);
  free(e);

  e = sox_create_effect(sox_find_effect("channels"));
  assert(sox_effect_options(e, 0, NULL) == SOX_SUCCESS);
  assert(sox_add_effect(chain, e, &input->signal, &output->signal) ==
         SOX_SUCCESS);
  free(e);

  e = sox_create_effect(sox_find_effect("output"));
  sox_args[0] = (char*)output;
  assert(sox_effect_options(e, 1, sox_args) == SOX_SUCCESS);
  assert(sox_add_effect(chain, e, &input->signal, &output->signal) ==
         SOX_SUCCESS);
  free(e);

  // Finally run the effects chain
  sox_flow_effects(chain, NULL, NULL);
  sox_delete_effects_chain(chain);
  sox_close(output);
  sox_close(input);

  // Pass audio to DeepSpeech
  char* result = DsSTT(ctx, (const short*)buffer, buffer_size / 2, sampleRate);
  if (result) {
    printf("%s\n", result);
  }
  free(result);

  // Deinitialise and quit
  DsClose(ctx);
  sox_quit();

  return 0;
}
