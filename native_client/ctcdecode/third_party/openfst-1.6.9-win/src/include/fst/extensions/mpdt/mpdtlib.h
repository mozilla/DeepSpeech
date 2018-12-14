// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// This is an experimental multipush-down transducer (MPDT) library. An MPDT is
// encoded as an FST, where some transitions are labeled with open or close
// parentheses, each mated pair of which is associated to one stack. To be
// interpreted as an MPDT, the parentheses within a stack must balance on a
// path.

#ifndef FST_EXTENSIONS_MPDT_MPDTLIB_H_
#define FST_EXTENSIONS_MPDT_MPDTLIB_H_

#include <fst/extensions/mpdt/compose.h>
#include <fst/extensions/mpdt/expand.h>
#include <fst/extensions/mpdt/mpdt.h>
#include <fst/extensions/mpdt/reverse.h>

#endif  // FST_EXTENSIONS_MPDT_MPDTLIB_H_
