#include <string>
#include <vector>
#include <fstream>
#include <unordered_set>
#include <iostream>
using namespace std;

#include "absl/types/optional.h"
#include "boost/program_options.hpp"

#include "ctcdecode/decoder_utils.h"
#include "ctcdecode/scorer.h"
#include "alphabet.h"
#include "deepspeech.h"

namespace po = boost::program_options;

int
create_package(absl::optional<string> alphabet_path,
               string lm_path,
               string vocab_path,
               string package_path,
               absl::optional<bool> force_bytes_output_mode,
               float default_alpha,
               float default_beta)
{
    // Read vocabulary
    unordered_set<string> words;
    bool vocab_looks_char_based = true;
    ifstream fin(vocab_path);
    if (!fin) {
        cerr << "Invalid vocabulary file " << vocab_path << "\n";
        return 1;
    }
    string word;
    while (fin >> word) {
        words.insert(word);
        if (get_utf8_str_len(word) > 1) {
            vocab_looks_char_based = false;
        }
    }
    cerr << words.size() << " unique words read from vocabulary file.\n"
         << (vocab_looks_char_based ? "Looks" : "Doesn't look")
         << " like a character based (Bytes Are All You Need) model.\n";

    if (!force_bytes_output_mode.has_value()) {
        force_bytes_output_mode = vocab_looks_char_based;
        cerr << "--force_bytes_output_mode was not specified, using value "
             << "infered from vocabulary contents: "
             << (vocab_looks_char_based ? "true" : "false") << "\n";
    }

    if (!force_bytes_output_mode.value() && !alphabet_path.has_value()) {
        cerr << "No --alphabet file specified, not using bytes output mode, can't continue.\n";
        return 1;
    }

    Scorer scorer;
    if (force_bytes_output_mode.value()) {
        scorer.set_alphabet(UTF8Alphabet());
    } else {
        Alphabet alphabet;
        alphabet.init(alphabet_path->c_str());
        scorer.set_alphabet(alphabet);
    }
    scorer.set_utf8_mode(force_bytes_output_mode.value());
    scorer.reset_params(default_alpha, default_beta);
    int err = scorer.load_lm(lm_path);
    if (err != DS_ERR_SCORER_NO_TRIE) {
        cerr << "Error loading language model file: "
             << DS_ErrorCodeToErrorMessage(err) << "\n";
        return 1;
    }
    scorer.fill_dictionary(words);

    // Copy LM file to final package file destination
    {
        ifstream lm_src(lm_path, std::ios::binary);
        ofstream package_dest(package_path, std::ios::binary);
        package_dest << lm_src.rdbuf();
    }

    // Save dictionary to package file, appending instead of overwriting
    if (!scorer.save_dictionary(package_path, true)) {
        cerr << "Error when saving package in " << package_path << ".\n";
        return 1;
    }

    cerr << "Package created in " << package_path << ".\n";
    return 0;
}

int
main(int argc, char** argv)
{
    po::options_description desc("Options");
    desc.add_options()
        ("help", "show help message")
        ("alphabet", po::value<string>(), "Path of alphabet file to use for vocabulary construction. Words with characters not in the alphabet will not be included in the vocabulary. Optional if using bytes output mode.")
        ("lm", po::value<string>(), "Path of KenLM binary LM file. Must be built without including the vocabulary (use the -v flag). See generate_lm.py for how to create a binary LM.")
        ("vocab", po::value<string>(), "Path of vocabulary file. Must contain words separated by whitespace.")
        ("package", po::value<string>(), "Path to save scorer package.")
        ("default_alpha", po::value<float>(), "Default value of alpha hyperparameter (float).")
        ("default_beta", po::value<float>(), "Default value of beta hyperparameter (float).")
        ("force_bytes_output_mode", po::value<bool>(), "Boolean flag, force set or unset bytes output mode in the scorer package. If not set, infers from the vocabulary. See <https://deepspeech.readthedocs.io/en/master/Decoder.html#bytes-output-mode> for further explanation.")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    // Check required flags.
    for (const string& flag : {"lm", "vocab", "package", "default_alpha", "default_beta"}) {
        if (!vm.count(flag)) {
            cerr << "--" << flag << " is a required flag. Pass --help for help.\n";
            return 1;
        }
    }

    // Parse optional --force_bytes_output_mode
    absl::optional<bool> force_bytes_output_mode = absl::nullopt;
    if (vm.count("force_bytes_output_mode")) {
        force_bytes_output_mode = vm["force_bytes_output_mode"].as<bool>();
    }

    // Parse optional --alphabet
    absl::optional<string> alphabet = absl::nullopt;
    if (vm.count("alphabet")) {
        alphabet = vm["alphabet"].as<string>();
    }

    create_package(alphabet,
                   vm["lm"].as<string>(),
                   vm["vocab"].as<string>(),
                   vm["package"].as<string>(),
                   force_bytes_output_mode,
                   vm["default_alpha"].as<float>(),
                   vm["default_beta"].as<float>());

    return 0;
}
