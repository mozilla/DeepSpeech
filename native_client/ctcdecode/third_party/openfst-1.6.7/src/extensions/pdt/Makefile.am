AM_CPPFLAGS = -I$(srcdir)/../../include $(ICU_CPPFLAGS)

if HAVE_BIN
bin_PROGRAMS = pdtcompose pdtexpand pdtinfo pdtreplace pdtreverse \
               pdtshortestpath

LDADD = libfstpdtscript.la \
        ../../script/libfstscript.la \
        ../../lib/libfst.la -lm $(DL_LIBS)

pdtcompose_SOURCES = pdtcompose.cc

pdtexpand_SOURCES = pdtexpand.cc

pdtinfo_SOURCES = pdtinfo.cc

pdtreplace_SOURCES = pdtreplace.cc

pdtreverse_SOURCES = pdtreverse.cc

pdtshortestpath_SOURCES = pdtshortestpath.cc
endif

if HAVE_SCRIPT
lib_LTLIBRARIES = libfstpdtscript.la
libfstpdtscript_la_SOURCES = getters.cc pdtscript.cc
libfstpdtscript_la_LDFLAGS = -version-info 10:0:0
libfstpdtscript_la_LIBADD = ../../script/libfstscript.la \
                            ../../lib/libfst.la -lm $(DL_LIBS)
endif
