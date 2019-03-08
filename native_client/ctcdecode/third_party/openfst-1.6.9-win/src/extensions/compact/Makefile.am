AM_CPPFLAGS = -I$(srcdir)/../../include $(ICU_CPPFLAGS)

libfstdir = @libfstdir@
libfst_LTLIBRARIES = compact8_acceptor-fst.la compact8_string-fst.la compact8_unweighted-fst.la compact8_unweighted_acceptor-fst.la compact8_weighted_string-fst.la compact16_acceptor-fst.la compact16_string-fst.la compact16_unweighted-fst.la compact16_unweighted_acceptor-fst.la compact16_weighted_string-fst.la compact64_acceptor-fst.la compact64_string-fst.la compact64_unweighted-fst.la compact64_unweighted_acceptor-fst.la compact64_weighted_string-fst.la

lib_LTLIBRARIES = libfstcompact.la

libfstcompact_la_SOURCES = compact8_acceptor-fst.cc compact8_string-fst.cc compact8_unweighted-fst.cc compact8_unweighted_acceptor-fst.cc compact8_weighted_string-fst.cc compact16_acceptor-fst.cc compact16_string-fst.cc compact16_unweighted-fst.cc compact16_unweighted_acceptor-fst.cc compact16_weighted_string-fst.cc compact64_acceptor-fst.cc compact64_string-fst.cc compact64_unweighted-fst.cc compact64_unweighted_acceptor-fst.cc compact64_weighted_string-fst.cc
libfstcompact_la_LDFLAGS = -version-info 13:0:0
libfstcompact_la_LIBADD = ../../lib/libfst.la -lm $(DL_LIBS)

compact8_acceptor_fst_la_SOURCES = compact8_acceptor-fst.cc
compact8_acceptor_fst_la_LDFLAGS = -module

compact8_string_fst_la_SOURCES = compact8_string-fst.cc
compact8_string_fst_la_LDFLAGS = -module

compact8_unweighted_fst_la_SOURCES = compact8_unweighted-fst.cc
compact8_unweighted_fst_la_LDFLAGS = -module

compact8_unweighted_acceptor_fst_la_SOURCES = compact8_unweighted_acceptor-fst.cc
compact8_unweighted_acceptor_fst_la_LDFLAGS = -module

compact8_weighted_string_fst_la_SOURCES = compact8_weighted_string-fst.cc
compact8_weighted_string_fst_la_LDFLAGS = -module

compact16_acceptor_fst_la_SOURCES = compact16_acceptor-fst.cc
compact16_acceptor_fst_la_LDFLAGS = -module

compact16_string_fst_la_SOURCES = compact16_string-fst.cc
compact16_string_fst_la_LDFLAGS = -module

compact16_unweighted_fst_la_SOURCES = compact16_unweighted-fst.cc
compact16_unweighted_fst_la_LDFLAGS = -module

compact16_unweighted_acceptor_fst_la_SOURCES = compact16_unweighted_acceptor-fst.cc
compact16_unweighted_acceptor_fst_la_LDFLAGS = -module

compact16_weighted_string_fst_la_SOURCES = compact16_weighted_string-fst.cc
compact16_weighted_string_fst_la_LDFLAGS = -module

compact64_acceptor_fst_la_SOURCES = compact64_acceptor-fst.cc
compact64_acceptor_fst_la_LDFLAGS = -module

compact64_string_fst_la_SOURCES = compact64_string-fst.cc
compact64_string_fst_la_LDFLAGS = -module

compact64_unweighted_fst_la_SOURCES = compact64_unweighted-fst.cc
compact64_unweighted_fst_la_LDFLAGS = -module

compact64_unweighted_acceptor_fst_la_SOURCES = compact64_unweighted_acceptor-fst.cc
compact64_unweighted_acceptor_fst_la_LDFLAGS = -module

compact64_weighted_string_fst_la_SOURCES = compact64_weighted_string-fst.cc
compact64_weighted_string_fst_la_LDFLAGS = -module
