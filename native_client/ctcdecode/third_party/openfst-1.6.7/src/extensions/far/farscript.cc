// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Definitions of 'scriptable' versions of FAR operations, that is,
// those that can be called with FstClass-type arguments.

#include <fst/extensions/far/farscript.h>
#include <fst/extensions/far/far.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

void FarCompileStrings(const std::vector<string> &in_fnames,
                       const string &out_fname, const string &arc_type,
                       const string &fst_type, const FarType &far_type,
                       int32 generate_keys, FarEntryType fet, FarTokenType tt,
                       const string &symbols_fname,
                       const string &unknown_symbol, bool keep_symbols,
                       bool initial_symbols, bool allow_negative_labels,
                       const string &key_prefix, const string &key_suffix) {
  FarCompileStringsArgs args(in_fnames, out_fname, fst_type, far_type,
                             generate_keys, fet, tt, symbols_fname,
                             unknown_symbol, keep_symbols, initial_symbols,
                             allow_negative_labels, key_prefix, key_suffix);
  Apply<Operation<FarCompileStringsArgs>>("FarCompileStrings", arc_type, &args);
}

void FarCreate(const std::vector<string> &in_fnames, const string &out_fname,
               const string &arc_type, const int32 generate_keys,
               const FarType &far_type, const string &key_prefix,
               const string &key_suffix) {
  FarCreateArgs args(in_fnames, out_fname, generate_keys, far_type, key_prefix,
                     key_suffix);
  Apply<Operation<FarCreateArgs>>("FarCreate", arc_type, &args);
}

bool FarEqual(const string &filename1, const string &filename2,
              const string &arc_type, float delta, const string &begin_key,
              const string &end_key) {
  FarEqualInnerArgs args(filename1, filename2, delta, begin_key, end_key);
  FarEqualArgs args_with_retval(args);
  Apply<Operation<FarEqualArgs>>("FarEqual", arc_type, &args_with_retval);
  return args_with_retval.retval;
}

void FarExtract(const std::vector<string> &ifilenames, const string &arc_type,
                int32 generate_filenames, const string &keys,
                const string &key_separator, const string &range_delimiter,
                const string &filename_prefix, const string &filename_suffix) {
  FarExtractArgs args(ifilenames, generate_filenames, keys, key_separator,
                      range_delimiter, filename_prefix, filename_suffix);
  Apply<Operation<FarExtractArgs>>("FarExtract", arc_type, &args);
}

void FarInfo(const std::vector<string> &filenames, const string &arc_type,
             const string &begin_key, const string &end_key,
             const bool list_fsts) {
  FarInfoArgs args(filenames, begin_key, end_key, list_fsts);
  Apply<Operation<FarInfoArgs>>("FarInfo", arc_type, &args);
}

void GetFarInfo(const std::vector<string> &filenames, const string &arc_type,
                const string &begin_key, const string &end_key,
                const bool list_fsts, FarInfoData *data) {
  GetFarInfoArgs args(filenames, begin_key, end_key, list_fsts, data);
  Apply<Operation<GetFarInfoArgs>>("GetFarInfo", arc_type, &args);
}

bool FarIsomorphic(const string &filename1, const string &filename2,
                   const string &arc_type, float delta, const string &begin_key,
                   const string &end_key) {
  FarIsomorphicInnerArgs args(filename1, filename2, delta, begin_key, end_key);
  FarIsomorphicArgs args_with_retval(args);
  Apply<Operation<FarIsomorphicArgs>>("FarIsomorphic", arc_type,
                                      &args_with_retval);
  return args_with_retval.retval;
}

void FarPrintStrings(const std::vector<string> &ifilenames,
                     const string &arc_type, const FarEntryType entry_type,
                     const FarTokenType token_type, const string &begin_key,
                     const string &end_key, const bool print_key,
                     const bool print_weight, const string &symbols_fname,
                     const bool initial_symbols, const int32 generate_filenames,
                     const string &filename_prefix,
                     const string &filename_suffix) {
  FarPrintStringsArgs args(ifilenames, entry_type, token_type, begin_key,
                           end_key, print_key, print_weight, symbols_fname,
                           initial_symbols, generate_filenames, filename_prefix,
                           filename_suffix);
  Apply<Operation<FarPrintStringsArgs>>("FarPrintStrings", arc_type, &args);
}

// Instantiate all templates for common arc types.

REGISTER_FST_FAR_OPERATIONS(StdArc);
REGISTER_FST_FAR_OPERATIONS(LogArc);
REGISTER_FST_FAR_OPERATIONS(Log64Arc);

}  // namespace script
}  // namespace fst
