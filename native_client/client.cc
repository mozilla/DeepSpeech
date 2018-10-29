#include <stdlib.h>
#include <stdio.h>

#include <assert.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <string.h>
#include <sox.h>
#include <time.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/stat.h>

#include <sstream>
#include <string>

#include "deepspeech.h"
#include "args.h"

#define N_CEP 26
#define N_CONTEXT 9
#define BEAM_WIDTH 500
#define LM_WEIGHT 1.50f
#define VALID_WORD_COUNT_WEIGHT 2.10f

typedef struct {
  const char* string;
  double cpu_time_overall;
} ds_result;

ds_result
LocalDsSTT(ModelState* aCtx, const short* aBuffer, size_t aBufferSize,
           int aSampleRate)
{
  ds_result res = {0};

  clock_t ds_start_time = clock();

  res.string = DS_SpeechToText(aCtx, aBuffer, aBufferSize, aSampleRate);

  clock_t ds_end_infer = clock();

  res.cpu_time_overall =
    ((double) (ds_end_infer - ds_start_time)) / CLOCKS_PER_SEC;

  return res;
}

typedef struct {
  char*  buffer;
  size_t buffer_size;
  int    sample_rate;
} ds_audio_buffer;

ds_audio_buffer
GetAudioBuffer(const char* path)
{
  ds_audio_buffer res = {0};

  sox_format_t* input = sox_open_read(path, NULL, NULL, NULL);
  assert(input);

  // Resample/reformat the audio so we can pass it through the MFCC functions
  sox_signalinfo_t target_signal = {
      16000, // Rate
      1, // Channels
      16, // Precision
      SOX_UNSPEC, // Length
      NULL // Effects headroom multiplier
  };

  sox_signalinfo_t interm_signal;

  sox_encodinginfo_t target_encoding = {
    SOX_ENCODING_SIGN2, // Sample format
    16, // Bits per sample
    0.0, // Compression factor
    sox_option_default, // Should bytes be reversed
    sox_option_default, // Should nibbles be reversed
    sox_option_default, // Should bits be reversed (pairs of bits?)
    sox_false // Reverse endianness
  };

#ifdef __APPLE__
  // It would be preferable to use sox_open_memstream_write here, but OS-X
  // doesn't support POSIX 2008, which it requires. See Issue #461.
  // Instead, we write to a temporary file.
  char* output_name = tmpnam(NULL);
  assert(output_name);
  sox_format_t* output = sox_open_write(output_name, &target_signal,
                                        &target_encoding, "raw", NULL, NULL);
#else
  sox_format_t* output = sox_open_memstream_write(&res.buffer,
                                                  &res.buffer_size,
                                                  &target_signal,
                                                  &target_encoding,
                                                  "raw", NULL);
#endif

  assert(output);

  res.sample_rate = (int)output->signal.rate;

  if ((int)input->signal.rate < 16000) {
    fprintf(stderr, "Warning: original sample rate (%d) is lower than 16kHz. Up-sampling might produce erratic speech recognition.\n", (int)input->signal.rate);
  }

  // Setup the effects chain to decode/resample
  char* sox_args[10];
  sox_effects_chain_t* chain =
    sox_create_effects_chain(&input->encoding, &output->encoding);

  interm_signal = input->signal;

  sox_effect_t* e = sox_create_effect(sox_find_effect("input"));
  sox_args[0] = (char*)input;
  assert(sox_effect_options(e, 1, sox_args) == SOX_SUCCESS);
  assert(sox_add_effect(chain, e, &interm_signal, &input->signal) ==
         SOX_SUCCESS);
  free(e);

  e = sox_create_effect(sox_find_effect("rate"));
  assert(sox_effect_options(e, 0, NULL) == SOX_SUCCESS);
  assert(sox_add_effect(chain, e, &interm_signal, &output->signal) ==
         SOX_SUCCESS);
  free(e);

  e = sox_create_effect(sox_find_effect("channels"));
  assert(sox_effect_options(e, 0, NULL) == SOX_SUCCESS);
  assert(sox_add_effect(chain, e, &interm_signal, &output->signal) ==
         SOX_SUCCESS);
  free(e);

  e = sox_create_effect(sox_find_effect("output"));
  sox_args[0] = (char*)output;
  assert(sox_effect_options(e, 1, sox_args) == SOX_SUCCESS);
  assert(sox_add_effect(chain, e, &interm_signal, &output->signal) ==
         SOX_SUCCESS);
  free(e);

  // Finally run the effects chain
  sox_flow_effects(chain, NULL, NULL);
  sox_delete_effects_chain(chain);

  // Close sox handles
  sox_close(output);
  sox_close(input);

#ifdef __APPLE__
  res.buffer_size = (size_t)(output->olength * 2);
  res.buffer = (char*)malloc(sizeof(char) * res.buffer_size);
  FILE* output_file = fopen(output_name, "rb");
  assert(fread(res.buffer, sizeof(char), res.buffer_size, output_file) == res.buffer_size);
  fclose(output_file);
  unlink(output_name);
#endif

  return res;
}

void
ProcessFile(ModelState* context, const char* path, bool show_times)
{
  ds_audio_buffer audio = GetAudioBuffer(path);

  // Pass audio to DeepSpeech
  // We take half of buffer_size because buffer is a char* while
  // LocalDsSTT() expected a short*
  ds_result result = LocalDsSTT(context,
                                (const short*)audio.buffer,
                                audio.buffer_size / 2,
                                audio.sample_rate);
  free(audio.buffer);

  if (result.string) {
    printf("%s\n", result.string);
    free((void*)result.string);
  }

  if (show_times) {
    printf("cpu_time_overall=%.05f\n",
           result.cpu_time_overall);
  }
}

int
main(int argc, char **argv)
{
  if (!ProcessArgs(argc, argv)) {
    return 1;
  }

  // Initialise DeepSpeech
  ModelState* ctx;
  int status = DS_CreateModel(model, N_CEP, N_CONTEXT, alphabet, BEAM_WIDTH, &ctx);
  if (status != 0) {
    fprintf(stderr, "Could not create model.\n");
    return 1;
  }

  if (lm && trie) {
    int status = DS_EnableDecoderWithLM(ctx,
                                        alphabet,
                                        lm,
                                        trie,
                                        LM_WEIGHT,
                                        VALID_WORD_COUNT_WEIGHT);
    if (status != 0) {
      fprintf(stderr, "Could not enable CTC decoder with LM.\n");
      return 1;
    }
  }

  // Initialise SOX
  assert(sox_init() == SOX_SUCCESS);

  struct stat wav_info;
  if (0 != stat(audio, &wav_info)) {
    printf("Error on stat: %d\n", errno);
  }

  switch (wav_info.st_mode & S_IFMT) {
    case S_IFLNK:
    case S_IFREG:
        ProcessFile(ctx, audio, show_times);
      break;

    case S_IFDIR:
        {
          printf("Running on directory %s\n", audio);
          DIR* wav_dir = opendir(audio);
          assert(wav_dir);

          struct dirent* entry;
          while ((entry = readdir(wav_dir)) != NULL) {
            std::string fname = std::string(entry->d_name);
            if (fname.find(".wav") == std::string::npos) {
              continue;
            }

            std::ostringstream fullpath;
            fullpath << audio << "/" << fname;
            std::string path = fullpath.str();
            printf("> %s\n", path.c_str());
            ProcessFile(ctx, path.c_str(), show_times);
          }
          closedir(wav_dir);
        }
      break;

    default:
        printf("Unexpected type for %s: %d\n", audio, (wav_info.st_mode & S_IFMT));
      break;
  }

  // Deinitialise and quit
  sox_quit();

  DS_DestroyModel(ctx);

  return 0;
}
