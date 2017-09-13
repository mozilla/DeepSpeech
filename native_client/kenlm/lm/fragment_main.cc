#include "lm/binary_format.hh"
#include "lm/model.hh"
#include "lm/left.hh"
#include "util/tokenize_piece.hh"

template <class Model> void Query(const char *name) {
  Model model(name);
  std::string line;
  lm::ngram::ChartState ignored;
  while (getline(std::cin, line)) {
    lm::ngram::RuleScore<Model> scorer(model, ignored);
    for (util::TokenIter<util::SingleCharacter, true> i(line, ' '); i; ++i) {
      scorer.Terminal(model.GetVocabulary().Index(*i));
    }
    std::cout << scorer.Finish() << '\n';
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Expected model file name." << std::endl;
    return 1;
  }
  const char *name = argv[1];
  lm::ngram::ModelType model_type = lm::ngram::PROBING;
  lm::ngram::RecognizeBinary(name, model_type);
  switch (model_type) {
    case lm::ngram::PROBING:
      Query<lm::ngram::ProbingModel>(name);
      break;
    case lm::ngram::REST_PROBING:
      Query<lm::ngram::RestProbingModel>(name);
      break;
    default:
      std::cerr << "Model type not supported yet." << std::endl;
  }
}
