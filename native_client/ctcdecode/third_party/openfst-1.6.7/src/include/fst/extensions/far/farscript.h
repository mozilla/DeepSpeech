// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Convenience file for including all of the FAR operations, or registering
// them for new arc types.

#ifndef FST_EXTENSIONS_FAR_FARSCRIPT_H_
#define FST_EXTENSIONS_FAR_FARSCRIPT_H_

#include <string>
#include <vector>

#include <fst/types.h>
#include <fst/extensions/far/compile-strings.h>
#include <fst/extensions/far/create.h>
#include <fst/extensions/far/equal.h>
#include <fst/extensions/far/extract.h>
#include <fst/extensions/far/far.h>
#include <fst/extensions/far/far-class.h>
#include <fst/extensions/far/info.h>
#include <fst/extensions/far/isomorphic.h>
#include <fst/extensions/far/print-strings.h>
#include <fst/extensions/far/script-impl.h>
#include <fst/script/arg-packs.h>

namespace fst {
namespace script {

// Note: it is safe to pass these strings as references because this struct is
// only used to pass them deeper in the call graph. Be sure you understand why
// this is so before using this struct for anything else!
struct FarCompileStringsArgs {
  const std::vector<string> &in_fnames;
  const string &out_fname;
  const string &fst_type;
  const FarType &far_type;
  const int32 generate_keys;
  const FarEntryType fet;
  const FarTokenType tt;
  const string &symbols_fname;
  const string &unknown_symbol;
  const bool keep_symbols;
  const bool initial_symbols;
  const bool allow_negative_labels;
  const string &key_prefix;
  const string &key_suffix;

  FarCompileStringsArgs(const std::vector<string> &in_fnames,
                        const string &out_fname, const string &fst_type,
                        const FarType &far_type, int32 generate_keys,
                        FarEntryType fet, FarTokenType tt,
                        const string &symbols_fname,
                        const string &unknown_symbol, bool keep_symbols,
                        bool initial_symbols, bool allow_negative_labels,
                        const string &key_prefix, const string &key_suffix)
      : in_fnames(in_fnames),
        out_fname(out_fname),
        fst_type(fst_type),
        far_type(far_type),
        generate_keys(generate_keys),
        fet(fet),
        tt(tt),
        symbols_fname(symbols_fname),
        unknown_symbol(unknown_symbol),
        keep_symbols(keep_symbols),
        initial_symbols(initial_symbols),
        allow_negative_labels(allow_negative_labels),
        key_prefix(key_prefix),
        key_suffix(key_suffix) {}
};

template <class Arc>
void FarCompileStrings(FarCompileStringsArgs *args) {
  FarCompileStrings<Arc>(
      args->in_fnames, args->out_fname, args->fst_type, args->far_type,
      args->generate_keys, args->fet, args->tt, args->symbols_fname,
      args->unknown_symbol, args->keep_symbols, args->initial_symbols,
      args->allow_negative_labels, args->key_prefix, args->key_suffix);
}

void FarCompileStrings(const std::vector<string> &in_fnames,
                       const string &out_fname, const string &arc_type,
                       const string &fst_type, const FarType &far_type,
                       int32 generate_keys, FarEntryType fet, FarTokenType tt,
                       const string &symbols_fname,
                       const string &unknown_symbol, bool keep_symbols,
                       bool initial_symbols, bool allow_negative_labels,
                       const string &key_prefix, const string &key_suffix);

// Note: it is safe to pass these strings as references because this struct is
// only used to pass them deeper in the call graph. Be sure you understand why
// this is so before using this struct for anything else!
struct FarCreateArgs {
  const std::vector<string> &in_fnames;
  const string &out_fname;
  const int32 generate_keys;
  const FarType &far_type;
  const string &key_prefix;
  const string &key_suffix;

  FarCreateArgs(const std::vector<string> &in_fnames, const string &out_fname,
                const int32 generate_keys, const FarType &far_type,
                const string &key_prefix, const string &key_suffix)
      : in_fnames(in_fnames),
        out_fname(out_fname),
        generate_keys(generate_keys),
        far_type(far_type),
        key_prefix(key_prefix),
        key_suffix(key_suffix) {}
};

template <class Arc>
void FarCreate(FarCreateArgs *args) {
  FarCreate<Arc>(args->in_fnames, args->out_fname, args->generate_keys,
                 args->far_type, args->key_prefix, args->key_suffix);
}

void FarCreate(const std::vector<string> &in_fnames, const string &out_fname,
               const string &arc_type, const int32 generate_keys,
               const FarType &far_type, const string &key_prefix,
               const string &key_suffix);

using FarEqualInnerArgs = std::tuple<const string &, const string &, float,
                                     const string &, const string &>;

using FarEqualArgs = WithReturnValue<bool, FarEqualInnerArgs>;

template <class Arc>
void FarEqual(FarEqualArgs *args) {
  args->retval = fst::FarEqual<Arc>(
      std::get<0>(args->args), std::get<1>(args->args), std::get<2>(args->args),
      std::get<3>(args->args), std::get<4>(args->args));
}

bool FarEqual(const string &filename1, const string &filename2,
              const string &arc_type, float delta = kDelta,
              const string &begin_key = string(),
              const string &end_key = string());

using FarExtractArgs =
    std::tuple<const std::vector<string> &, int32, const string &,
               const string &, const string &, const string &, const string &>;

template <class Arc>
void FarExtract(FarExtractArgs *args) {
  fst::FarExtract<Arc>(std::get<0>(*args), std::get<1>(*args),
                           std::get<2>(*args), std::get<3>(*args),
                           std::get<4>(*args), std::get<5>(*args),
                           std::get<6>(*args));
}

void FarExtract(const std::vector<string> &ifilenames, const string &arc_type,
                int32 generate_filenames, const string &keys,
                const string &key_separator, const string &range_delimiter,
                const string &filename_prefix, const string &filename_suffix);

using FarInfoArgs = std::tuple<const std::vector<string> &, const string &,
                               const string &, const bool>;

template <class Arc>
void FarInfo(FarInfoArgs *args) {
  fst::FarInfo<Arc>(std::get<0>(*args), std::get<1>(*args),
                        std::get<2>(*args), std::get<3>(*args));
}

void FarInfo(const std::vector<string> &filenames, const string &arc_type,
             const string &begin_key, const string &end_key,
             const bool list_fsts);

using GetFarInfoArgs = std::tuple<const std::vector<string> &, const string &,
                                  const string &, const bool, FarInfoData *>;

template <class Arc>
void GetFarInfo(GetFarInfoArgs *args) {
  fst::GetFarInfo<Arc>(std::get<0>(*args), std::get<1>(*args),
                           std::get<2>(*args), std::get<3>(*args),
                           std::get<4>(*args));
}

void GetFarInfo(const std::vector<string> &filenames, const string &arc_type,
                const string &begin_key, const string &end_key,
                const bool list_fsts, FarInfoData *);

using FarIsomorphicInnerArgs = std::tuple<const string &, const string &, float,
                                          const string &, const string &>;

using FarIsomorphicArgs = WithReturnValue<bool, FarIsomorphicInnerArgs>;

template <class Arc>
void FarIsomorphic(FarIsomorphicArgs *args) {
  args->retval = fst::FarIsomorphic<Arc>(
      std::get<0>(args->args), std::get<1>(args->args), std::get<2>(args->args),
      std::get<3>(args->args), std::get<4>(args->args));
}

bool FarIsomorphic(const string &filename1, const string &filename2,
                   const string &arc_type, float delta = kDelta,
                   const string &begin_key = string(),
                   const string &end_key = string());

struct FarPrintStringsArgs {
  const std::vector<string> &ifilenames;
  const FarEntryType entry_type;
  const FarTokenType token_type;
  const string &begin_key;
  const string &end_key;
  const bool print_key;
  const bool print_weight;
  const string &symbols_fname;
  const bool initial_symbols;
  const int32 generate_filenames;
  const string &filename_prefix;
  const string &filename_suffix;

  FarPrintStringsArgs(const std::vector<string> &ifilenames,
                      const FarEntryType entry_type,
                      const FarTokenType token_type, const string &begin_key,
                      const string &end_key, const bool print_key,
                      const bool print_weight, const string &symbols_fname,
                      const bool initial_symbols,
                      const int32 generate_filenames,
                      const string &filename_prefix,
                      const string &filename_suffix)
      : ifilenames(ifilenames),
        entry_type(entry_type),
        token_type(token_type),
        begin_key(begin_key),
        end_key(end_key),
        print_key(print_key),
        print_weight(print_weight),
        symbols_fname(symbols_fname),
        initial_symbols(initial_symbols),
        generate_filenames(generate_filenames),
        filename_prefix(filename_prefix),
        filename_suffix(filename_suffix) {}
};

template <class Arc>
void FarPrintStrings(FarPrintStringsArgs *args) {
  fst::FarPrintStrings<Arc>(
      args->ifilenames, args->entry_type, args->token_type, args->begin_key,
      args->end_key, args->print_key, args->print_weight, args->symbols_fname,
      args->initial_symbols, args->generate_filenames, args->filename_prefix,
      args->filename_suffix);
}

void FarPrintStrings(const std::vector<string> &ifilenames,
                     const string &arc_type, const FarEntryType entry_type,
                     const FarTokenType token_type, const string &begin_key,
                     const string &end_key, const bool print_key,
                     const bool print_weight, const string &symbols_fname,
                     const bool initial_symbols, const int32 generate_filenames,
                     const string &filename_prefix,
                     const string &filename_suffix);

}  // namespace script
}  // namespace fst

#define REGISTER_FST_FAR_OPERATIONS(ArcType)                                 \
  REGISTER_FST_OPERATION(FarCompileStrings, ArcType, FarCompileStringsArgs); \
  REGISTER_FST_OPERATION(FarCreate, ArcType, FarCreateArgs);                 \
  REGISTER_FST_OPERATION(FarEqual, ArcType, FarEqualArgs);                   \
  REGISTER_FST_OPERATION(FarExtract, ArcType, FarExtractArgs);               \
  REGISTER_FST_OPERATION(FarInfo, ArcType, FarInfoArgs);                     \
  REGISTER_FST_OPERATION(FarIsomorphic, ArcType, FarIsomorphicArgs);         \
  REGISTER_FST_OPERATION(FarPrintStrings, ArcType, FarPrintStringsArgs);     \
  REGISTER_FST_OPERATION(GetFarInfo, ArcType, GetFarInfoArgs)

#endif  // FST_EXTENSIONS_FAR_FARSCRIPT_H_
