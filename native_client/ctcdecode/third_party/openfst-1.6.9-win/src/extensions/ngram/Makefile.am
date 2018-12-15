AM_CPPFLAGS = -I$(srcdir)/../../include $(ICU_CPPFLAGS)

libfstdir = @libfstdir@
libfst_LTLIBRARIES = ngram-fst.la

lib_LTLIBRARIES = libfstngram.la

ngram_fst_la_SOURCES = bitmap-index.cc ngram-fst.cc nthbit.cc
ngram_fst_la_LDFLAGS = -module

libfstngram_la_SOURCES = bitmap-index.cc ngram-fst.cc nthbit.cc
libfstngram_la_LDFLAGS = -version-info 13:0:0
libfstngram_la_LIBADD = ../../lib/libfst.la -lm $(DL_LIBS)
