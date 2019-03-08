// Work-around to correctly build (e.g. distclean) with autotools
// using files in another directory that are also built there.
// See https://stackoverflow.com/questions/30379837.

#include "fstconvert-main.cc"   // NOLINT
#include "fstconvert.cc"        // NOLINT
