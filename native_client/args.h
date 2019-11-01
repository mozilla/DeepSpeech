#ifndef __ARGS_H__
#define __ARGS_H__

#if defined(_MSC_VER)
#include "getopt_win.h"
#else
#include <getopt.h>
#endif
#include <iostream>

#include "deepspeech.h"

char* model = NULL;

char* lm = NULL;

char* trie = NULL;

char* audio = NULL;

int beam_width = 500;

float lm_alpha = 0.75f;

float lm_beta = 1.85f;

bool load_without_trie = false;

bool show_times = false;

bool has_versions = false;

bool extended_metadata = false;

bool json_output = false;

int stream_size = 0;

void PrintHelp(const char* bin)
{
    std::cout <<
    "Usage: " << bin << " --model MODEL [--lm LM --trie TRIE] --audio AUDIO [-t] [-e]\n"
    "\n"
    "Running DeepSpeech inference.\n"
    "\n"
    "	--model MODEL		Path to the model (protocol buffer binary file)\n"
    "	--lm LM			Path to the language model binary file\n"
    "	--trie TRIE		Path to the language model trie file created with native_client/generate_trie\n"
    "	--audio AUDIO		Path to the audio file to run (WAV format)\n"
    "	--beam_width BEAM_WIDTH	Value for decoder beam width (int)\n"
    "	--lm_alpha LM_ALPHA	Value for language model alpha param (float)\n"
    "	--lm_beta LM_BETA	Value for language model beta param (float)\n"
    "	-t			Run in benchmark mode, output mfcc & inference time\n"
    "	--extended		Output string from extended metadata\n"
    "	--json			Extended output, shows word timings as JSON\n"
    "	--stream size		Run in stream mode, output intermediate results\n"
    "	--help			Show help\n"
    "	--version		Print version and exits\n";
    DS_PrintVersions();
    exit(1);
}

bool ProcessArgs(int argc, char** argv)
{
    const char* const short_opts = "m:a:l:r:w:c:d:b:tehv";
    const option long_opts[] = {
            {"model", required_argument, nullptr, 'm'},
            {"lm", required_argument, nullptr, 'l'},
            {"trie", required_argument, nullptr, 'r'},
            {"audio", required_argument, nullptr, 'w'},
            {"beam_width", required_argument, nullptr, 'b'},
            {"lm_alpha", required_argument, nullptr, 'c'},
            {"lm_beta", required_argument, nullptr, 'd'},
            {"run_very_slowly_without_trie_I_really_know_what_Im_doing", no_argument, nullptr, 999},
            {"t", no_argument, nullptr, 't'},
            {"extended", no_argument, nullptr, 'e'},
            {"json", no_argument, nullptr, 'j'},
            {"stream", required_argument, nullptr, 's'},
            {"help", no_argument, nullptr, 'h'},
            {"version", no_argument, nullptr, 'v'},
            {nullptr, no_argument, nullptr, 0}
    };

    while (true)
    {
        const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

        if (-1 == opt)
            break;

        switch (opt)
        {
        case 'm':
            model = optarg;
            break;

        case 'l':
            lm = optarg;
            break;

        case 'r':
            trie = optarg;
            break;

        case 'w':
            audio = optarg;
            break;

	case 'b':
	    beam_width = atoi(optarg);
	    break;
	
	case 'c':
	    lm_alpha = atof(optarg);
	    break;
	
	case 'd':
	    lm_beta = atof(optarg);
	    break;

        case 999:
            load_without_trie = true;
            break;

        case 't':
            show_times = true;
            break;

        case 'v':
            has_versions = true;
            break;

        case 'e':
            extended_metadata = true;
            break;

        case 'j':
            json_output = true;
            break;

        case 's':
            stream_size = atoi(optarg);
            break;

        case 'h': // -h or --help
        case '?': // Unrecognized option
        default:
            PrintHelp(argv[0]);
            break;
        }
    }

    if (has_versions) {
        DS_PrintVersions();
        return false;
    }

    if (!model || !audio) {
        PrintHelp(argv[0]);
        return false;
    }

    if (stream_size < 0 || stream_size % 160 != 0) {
        std::cout <<
        "Stream buffer size must be multiples of 160\n";
        return false;
    }

    return true;
}

#endif // __ARGS_H__
