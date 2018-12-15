AM_CPPFLAGS = -I$(srcdir)/../../include $(ICU_CPPFLAGS)

libfstdir = @libfstdir@
libfst_LTLIBRARIES = arc_lookahead-fst.la \
ilabel_lookahead-fst.la olabel_lookahead-fst.la

lib_LTLIBRARIES = libfstlookahead.la

libfstlookahead_la_SOURCES = arc_lookahead-fst.cc ilabel_lookahead-fst.cc \
                             olabel_lookahead-fst.cc
libfstlookahead_la_LDFLAGS = -version-info 13:0:0
libfstlookahead_la_LIBADD = ../../lib/libfst.la -lm $(DL_LIBS)

arc_lookahead_fst_la_SOURCES = arc_lookahead-fst.cc
arc_lookahead_fst_la_LDFLAGS = -module

ilabel_lookahead_fst_la_SOURCES = ilabel_lookahead-fst.cc
ilabel_lookahead_fst_la_LDFLAGS = -module

olabel_lookahead_fst_la_SOURCES = olabel_lookahead-fst.cc
olabel_lookahead_fst_la_LDFLAGS = -module
