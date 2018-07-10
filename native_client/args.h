#ifndef __ARGS_H__
#define __ARGS_H__

#include <getopt.h>
#include <iostream>

#include "deepspeech.h"

bool has_model = false;
std::string model;

bool has_alphabet = false;
std::string alphabet;

bool has_lm = false;
std::string lm;

bool has_trie = false;
std::string trie;

bool has_audio = false;
std::string audio;

bool show_times = false;

bool has_versions = false;

using namespace DeepSpeech;

void PrintHelp(const char* bin)
{
    std::cout <<
    "Usage: " << std::string(bin) << " --model MODEL --alphabet ALPHABET [--lm LM --trie TRIE] --audio AUDIO [-t]\n"
    "\n"
    "Running DeepSpeech inference.\n"
    "\n"
    "	--model MODEL		Path to the model (protocol buffer binary file)\n"
    "	--alphabet ALPHABET	Path to the configuration file specifying the alphabet used by the network\n"
    "	--lm LM			Path to the language model binary file\n"
    "	--trie TRIE		Path to the language model trie file created with native_client/generate_trie\n"
    "	--audio AUDIO		Path to the audio file to run (WAV format)\n"
    "	-t			Run in benchmark mode, output mfcc & inference time\n"
    "	--help			Show help\n"
    "	--version		Print version and exits\n";
    print_versions();
    exit(1);
}

bool ProcessArgs(int argc, char** argv)
{
    const char* const short_opts = "m:a:l:r:w:thv";
    const option long_opts[] = {
            {"model", required_argument, nullptr, 'm'},
            {"alphabet", required_argument, nullptr, 'a'},
            {"lm", required_argument, nullptr, 'l'},
            {"trie", required_argument, nullptr, 'r'},
            {"audio", required_argument, nullptr, 'w'},
            {"t", no_argument, nullptr, 't'},
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
            model     = std::string(optarg);
            has_model = true;
            break;

        case 'a':
            alphabet     = std::string(optarg);
            has_alphabet = true;
            break;

        case 'l':
            lm     = std::string(optarg);
            has_lm = true;
            break;

        case 'r':
            trie     = std::string(optarg);
            has_trie = true;
            break;

        case 'w':
            audio     = std::string(optarg);
            has_audio = true;
            break;

        case 't':
            show_times = true;
            break;

        case 'v':
            has_versions = true;
            break;

        case 'h': // -h or --help
        case '?': // Unrecognized option
        default:
            PrintHelp(argv[0]);
            break;
        }
    }

    if (has_versions) {
        print_versions();
        return false;
    }

    if (!has_model || !has_alphabet || !has_audio || alphabet.length() == 0 || audio.length() == 0) {
        PrintHelp(argv[0]);
        return false;
    }

    return true;
}

#endif // __ARGS_H__
