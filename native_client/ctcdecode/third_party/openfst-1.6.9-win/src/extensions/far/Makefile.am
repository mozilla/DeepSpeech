AM_CPPFLAGS = -I$(srcdir)/../../include $(ICU_CPPFLAGS)

if HAVE_SCRIPT
lib_LTLIBRARIES = libfstfar.la libfstfarscript.la
else
lib_LTLIBRARIES = libfstfar.la
endif

libfstfar_la_SOURCES = sttable.cc stlist.cc
libfstfar_la_LDFLAGS = -version-info 13:0:0
libfstfar_la_LIBADD = ../../lib/libfst.la -lm $(DL_LIBS)

if HAVE_SCRIPT
libfstfarscript_la_SOURCES = far-class.cc farscript.cc getters.cc script-impl.cc \
                             strings.cc
libfstfarscript_la_LDFLAGS = -version-info 13:0:0
libfstfarscript_la_LIBADD = \
    libfstfar.la ../../script/libfstscript.la \
        ../../lib/libfst.la -lm $(DL_LIBS)
endif

if HAVE_BIN
bin_PROGRAMS = farcompilestrings farcreate farequal farextract farinfo \
    farisomorphic farprintstrings

LDADD = libfstfarscript.la ../../script/libfstscript.la \
        ../../lib/libfst.la -lm $(DL_LIBS)

farcompilestrings_SOURCES = farcompilestrings.cc

farcreate_SOURCES = farcreate.cc

farequal_SOURCES = farequal.cc

farextract_SOURCES = farextract.cc

farinfo_SOURCES = farinfo.cc

farisomorphic_SOURCES = farisomorphic.cc

farprintstrings_SOURCES = farprintstrings.cc
endif
