AM_CPPFLAGS = -I$(srcdir)/../include $(ICU_CPPFLAGS)

lib_LTLIBRARIES = libfst.la
libfst_la_SOURCES = compat.cc flags.cc fst.cc fst-types.cc mapped-file.cc \
                    properties.cc symbol-table.cc symbol-table-ops.cc \
                    weight.cc util.cc
libfst_la_LDFLAGS = -version-info 10:0:0
libfst_la_LIBADD = $(DL_LIBS)
