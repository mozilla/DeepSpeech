#!/bin/bash
#This is just an example compilation.  You should integrate these files into your build system.  Boost jam is provided and preferred.

echo You must use ./bjam if you want language model estimation, filtering, or support for compressed files \(.gz, .bz2, .xz\) 1>&2

rm {lm,util}/*.o 2>/dev/null
set -e

CXX=${CXX:-g++}

CXXFLAGS+=" -I. -O3 -DNDEBUG -DKENLM_MAX_ORDER=6"

#If this fails for you, consider using bjam.
if [ ${#NPLM} != 0 ]; then
  CXXFLAGS+=" -DHAVE_NPLM -lneuralLM -L$NPLM/src -I$NPLM/src -lboost_thread-mt -fopenmp"
  ADDED_PATHS="lm/wrappers/*.cc"
fi
echo 'Compiling with '$CXX $CXXFLAGS

#Grab all cc files in these directories except those ending in test.cc or main.cc
objects=""
for i in util/double-conversion/*.cc util/*.cc lm/*.cc $ADDED_PATHS; do
  if [ "${i%test.cc}" == "$i" ] && [ "${i%main.cc}" == "$i" ]; then
    $CXX $CXXFLAGS -c $i -o ${i%.cc}.o
    objects="$objects ${i%.cc}.o"
  fi
done

mkdir -p bin
if [ "$(uname)" != Darwin ]; then
  CXXFLAGS="$CXXFLAGS -lrt"
fi
$CXX lm/build_binary_main.cc $objects -o bin/build_binary $CXXFLAGS $LDFLAGS
$CXX lm/query_main.cc $objects -o bin/query $CXXFLAGS $LDFLAGS
