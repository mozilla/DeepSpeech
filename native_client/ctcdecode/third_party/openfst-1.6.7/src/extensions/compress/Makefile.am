AM_CPPFLAGS = -I$(srcdir)/../../include $(ICU_CPPFLAGS)

if HAVE_BIN
bin_PROGRAMS = fstcompress fstrandmod

LDADD = libfstcompressscript.la \
        ../../script/libfstscript.la \
        ../../lib/libfst.la \
        -lm $(DL_LIBS)

fstcompress_SOURCES = fstcompress.cc
fstrandmod_SOURCES = fstrandmod.cc
endif

if HAVE_SCRIPT
libfstcompressscript_la_SOURCES = compress-script.cc
libfstcompressscript_la_LDFLAGS = -version-info 10:0:0
libfstcompressscript_la_LIBADD = \
        ../../script/libfstscript.la \
        ../../lib/libfst.la -lz -lm $(DL_LIBS)
endif

if HAVE_SCRIPT
lib_LTLIBRARIES = libfstcompressscript.la
endif
