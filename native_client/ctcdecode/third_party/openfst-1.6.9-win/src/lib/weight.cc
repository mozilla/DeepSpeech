#include <fst/weight.h>

DEFINE_string(fst_weight_separator, ",",
              "Character separator between printed composite weights; "
              "must be a single character");

DEFINE_string(fst_weight_parentheses, "",
              "Characters enclosing the first weight of a printed composite "
              "weight (e.g., pair weight, tuple weight and derived classes) to "
              "ensure proper I/O of nested composite weights; "
              "must have size 0 (none) or 2 (open and close parenthesis)");

namespace fst {

namespace internal {

CompositeWeightIO::CompositeWeightIO(char separator,
                                     std::pair<char, char> parentheses)
    : separator_(separator),
      open_paren_(parentheses.first),
      close_paren_(parentheses.second),
      error_(false) {
  if ((open_paren_ == 0 || close_paren_ == 0) && open_paren_ != close_paren_) {
    FSTERROR() << "Invalid configuration of weight parentheses: "
               << static_cast<int>(open_paren_) << " "
               << static_cast<int>(close_paren_);
    error_ = true;
  }
}

CompositeWeightIO::CompositeWeightIO()
    : CompositeWeightIO(FLAGS_fst_weight_separator.empty()
                            ? 0
                            : FLAGS_fst_weight_separator.front(),
                        {FLAGS_fst_weight_parentheses.empty()
                             ? 0
                             : FLAGS_fst_weight_parentheses[0],
                         FLAGS_fst_weight_parentheses.size() < 2
                             ? 0
                             : FLAGS_fst_weight_parentheses[1]}) {
  if (FLAGS_fst_weight_separator.size() != 1) {
    FSTERROR() << "CompositeWeight: "
               << "FLAGS_fst_weight_separator.size() is not equal to 1";
    error_ = true;
  }
  if (!FLAGS_fst_weight_parentheses.empty() &&
      FLAGS_fst_weight_parentheses.size() != 2) {
    FSTERROR() << "CompositeWeight: "
               << "FLAGS_fst_weight_parentheses.size() is not equal to 2";
    error_ = true;
  }
}

}  // namespace internal

CompositeWeightWriter::CompositeWeightWriter(std::ostream &ostrm)
    : ostrm_(ostrm) {
  if (error()) ostrm.clear(std::ios::badbit);
}

CompositeWeightWriter::CompositeWeightWriter(std::ostream &ostrm,
                                             char separator,
                                             std::pair<char, char> parentheses)
    : internal::CompositeWeightIO(separator, parentheses), ostrm_(ostrm) {
  if (error()) ostrm_.clear(std::ios::badbit);
}

void CompositeWeightWriter::WriteBegin() {
  if (open_paren_ != 0) {
    ostrm_ << open_paren_;
  }
}

void CompositeWeightWriter::WriteEnd() {
  if (close_paren_ != 0) {
    ostrm_ << close_paren_;
  }
}

CompositeWeightReader::CompositeWeightReader(std::istream &istrm)
    : istrm_(istrm) {
  if (error()) istrm_.clear(std::ios::badbit);
}

CompositeWeightReader::CompositeWeightReader(std::istream &istrm,
                                             char separator,
                                             std::pair<char, char> parentheses)
    : internal::CompositeWeightIO(separator, parentheses), istrm_(istrm) {
  if (error()) istrm_.clear(std::ios::badbit);
}

void CompositeWeightReader::ReadBegin() {
  do {  // Skips whitespace.
    c_ = istrm_.get();
  } while (std::isspace(c_));
  if (open_paren_ != 0) {
    if (c_ != open_paren_) {
      FSTERROR() << "CompositeWeightReader: Open paren missing: "
                 << "fst_weight_parentheses flag set correcty?";
      istrm_.clear(std::ios::badbit);
      return;
    }
    ++depth_;
    c_ = istrm_.get();
  }
}

void CompositeWeightReader::ReadEnd() {
  if (c_ != EOF && !std::isspace(c_)) {
    FSTERROR() << "CompositeWeightReader: excess character: '"
               << static_cast<char>(c_)
               << "': fst_weight_parentheses flag set correcty?";
    istrm_.clear(std::ios::badbit);
  }
}

}  // namespace fst
