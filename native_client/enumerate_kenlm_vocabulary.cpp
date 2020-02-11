#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "lm/enumerate_vocab.hh"
#include "lm/virtual_interface.hh"
#include "lm/word_index.hh"
#include "lm/model.hh"

const std::string START_TOKEN = "<s>";
const std::string UNK_TOKEN = "<unk>";
const std::string END_TOKEN = "</s>";

// Implement a callback to retrieve the dictionary of language model.
class RetrieveStrEnumerateVocab : public lm::EnumerateVocab
{
public:
  RetrieveStrEnumerateVocab() {}

  void Add(lm::WordIndex index, const StringPiece &str) {
    vocabulary.push_back(std::string(str.data(), str.length()));
  }

  std::vector<std::string> vocabulary;
};

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <kenlm_model> <output_path>" << std::endl;
    return -1;
  }

  const char* kenlm_model = argv[1];
  const char* output_path = argv[2];

  std::unique_ptr<lm::base::Model> language_model_;
  lm::ngram::Config config;
  RetrieveStrEnumerateVocab enumerate;
  config.enumerate_vocab = &enumerate;
  language_model_.reset(lm::ngram::LoadVirtual(kenlm_model, config));

  std::ofstream fout(output_path);
  for (const std::string& word : enumerate.vocabulary) {
    fout << word << "\n";
  }

  return 0;
}
