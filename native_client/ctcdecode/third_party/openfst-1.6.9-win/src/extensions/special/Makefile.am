AM_CPPFLAGS = -I$(srcdir)/../../include -I$(srcdir)/../../bin $(ICU_CPPFLAGS)

if HAVE_BIN
bin_PROGRAMS = fstspecial

LDADD = ../../script/libfstscript.la \
        ../../lib/libfst.la -lm $(DL_LIBS)

fstspecial_SOURCES = fstspecial.cc phi-fst.cc rho-fst.cc sigma-fst.cc
fstspecial_CPPFLAGS = $(AM_CPPFLAGS)
endif

libfstdir = @libfstdir@
libfst_LTLIBRARIES = phi-fst.la rho-fst.la sigma-fst.la

lib_LTLIBRARIES = libfstspecial.la

libfstspecial_la_SOURCES = phi-fst.cc rho-fst.cc sigma-fst.cc
libfstspecial_la_LDFLAGS = -version-info 13:0:0 -lm $(DL_LIBS)
libfstspecial_la_LIBADD = ../../lib/libfst.la -lm $(DL_LIBS)

phi_fst_la_SOURCES = phi-fst.cc
phi_fst_la_LDFLAGS = -module
phi_fst_la_LIBADD = ../../lib/libfst.la -lm $(DL_LIBS)

rho_fst_la_SOURCES = rho-fst.cc
rho_fst_la_LDFLAGS = -module
rho_fst_la_LIBADD = ../../lib/libfst.la -lm $(DL_LIBS)

sigma_fst_la_SOURCES = sigma-fst.cc
sigma_fst_la_LDFLAGS = -module
sigma_fst_la_LIBADD = ../../lib/libfst.la -lm $(DL_LIBS)
