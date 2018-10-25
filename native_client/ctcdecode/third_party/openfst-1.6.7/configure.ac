AC_INIT([OpenFst], [1.6.7], [help@www.openfst.org])
AM_INIT_AUTOMAKE([foreign nostdinc -Wall -Werror subdir-objects])
AM_PROG_AR

# OpenFst does not throw exceptions, so we do not generate exception handling
# code. However, users are free to re-enable exception handling.
# OpenFst assumes char is unsigned; -fsigned-char is likely unsafe.
CPPFLAGS="$CPPFLAGS -fno-exceptions -funsigned-char"
CXXFLAGS="$CXXFLAGS -std=c++11"

AC_PROG_CXX
AC_DISABLE_STATIC
AC_PROG_LIBTOOL

AC_CONFIG_HEADERS([config.h src/include/fst/config.h])
AC_CONFIG_SRCDIR([src/lib/fst.cc])
AC_CONFIG_FILES([
  Makefile
  src/Makefile
  src/include/Makefile
  src/lib/Makefile
  src/bin/Makefile
  src/test/Makefile
  src/extensions/Makefile
  src/extensions/compact/Makefile
  src/extensions/compress/Makefile
  src/extensions/const/Makefile
  src/extensions/far/Makefile
  src/extensions/linear/Makefile
  src/extensions/lookahead/Makefile
  src/extensions/mpdt/Makefile
  src/extensions/ngram/Makefile
  src/extensions/pdt/Makefile
  src/extensions/python/Makefile
  src/extensions/special/Makefile
  src/script/Makefile
])
AC_CONFIG_MACRO_DIR([m4])
AC_LANG([C++])

AC_ARG_ENABLE([compact-fsts],
              [AS_HELP_STRING([--enable-compact-fsts],
              [enable CompactFst extensions])],
              [],
              [enable_compact_fsts=no])
AM_CONDITIONAL([HAVE_COMPACT], [test "x$enable_compact_fsts" != xno])

AC_ARG_ENABLE([compress],
              [AS_HELP_STRING([--enable-compress],
              [enable compression extension])],
              [],
              [enable_compress=no])
AM_CONDITIONAL([HAVE_COMPRESS], [test "x$enable_compress" != xno])

AC_ARG_ENABLE([const-fsts],
              [AS_HELP_STRING([--enable-const-fsts],
              [enable ConstFst extensions])],
              [],
              [enable_const_fsts=no])
AM_CONDITIONAL([HAVE_CONST], [test "x$enable_const_fsts" != xno])

AC_ARG_ENABLE([far],
              [AS_HELP_STRING([--enable-far], [enable FAR extensions])],
              [],
              [enable_far=no])
AM_CONDITIONAL([HAVE_FAR], [test "x$enable_far" != xno])

AC_ARG_ENABLE([linear-fsts],
              [AS_HELP_STRING([--enable-linear-fsts],
              [enable LinearTagger/ClassifierFst extensions])],
              [],
              [enable_linear_fsts=no])
AM_CONDITIONAL([HAVE_LINEAR], [test "x$enable_linear_fsts" != xno])

AC_ARG_ENABLE([lookahead-fsts],
              [AS_HELP_STRING([--enable-lookahead-fsts],
              [enable LookAheadFst extensions])],
              [],
              [enable_lookahead_fsts=no])
AM_CONDITIONAL([HAVE_LOOKAHEAD], [test "x$enable_lookahead_fsts" != xno])

AC_ARG_ENABLE([mpdt],
              [AS_HELP_STRING([--enable-mpdt],
              [enable MPDT extensions])],
              [],
              [enable_mpdt=no])
AM_CONDITIONAL([HAVE_MPDT], [test "x$enable_mpdt" != xno])

AC_ARG_ENABLE([ngram-fsts],
              [AS_HELP_STRING([--enable-ngram-fsts],
              [enable NGramFst extension])],
              [],
              [enable_ngram_fsts=no])
AM_CONDITIONAL([HAVE_NGRAM], [test "x$enable_ngram_fsts" != xno])

AC_ARG_ENABLE([pdt],
              [AS_HELP_STRING([--enable-pdt],
              [enable PDT extensions])],
              [],
              [enable_pdt=no])
AM_CONDITIONAL([HAVE_PDT], [test "x$enable_pdt" != xno])

AC_ARG_ENABLE([python],
              [AS_HELP_STRING([--enable-python],
              [enable Python extensions])],
              [],
              [enable_python=no])
AM_CONDITIONAL([HAVE_PYTHON], [test "x$enable_python" != xno])
if test "x$enable_python" != xno; then
  AM_PATH_PYTHON(2.7)
  AC_PYTHON_DEVEL([>= '2.7'])
fi

AC_ARG_ENABLE([special],
              [AS_HELP_STRING([--enable-special],
              [enable special-matcher extensions])],
              [],
              [enable_special=no])
AM_CONDITIONAL([HAVE_SPECIAL], [test "x$enable_special" != xno])

# --enable-bin enables script and bin "extensions".
AC_ARG_ENABLE([bin],
              [AS_HELP_STRING([--enable-bin],
              [enable fst::script and command-line binaries])],
              [],
              [enable_bin=yes])
AM_CONDITIONAL([HAVE_BIN], [test "x$enable_bin" != xno])
AM_CONDITIONAL([HAVE_SCRIPT], [test "x$enable_bin" != xno])

# --enable-grm enables dependencies of OpenGrm: far, mpdt, and pdt.
AC_ARG_ENABLE([grm],
              [AS_HELP_STRING([--enable-grm],
              [enable all dependencies of OpenGrm])],
              [],
              [enable_grm=no])
AM_CONDITIONAL([HAVE_GRM], [test "x$enable_grm" != xno])

AC_ARG_WITH([libfstdir],
[--with-libfstdir[=DIR] fst dynamic extensions [[LIBDIR/fst]]],
[], [with_libfstdir=[${libdir}/fst]])

AC_SUBST([libfstdir], $with_libfstdir)

AC_CHECK_LIB([dl], dlopen, [DL_LIBS=-ldl])
AC_SUBST([DL_LIBS])

AC_OUTPUT
