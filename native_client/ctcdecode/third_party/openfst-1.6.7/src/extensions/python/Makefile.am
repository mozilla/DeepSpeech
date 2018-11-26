# NB: we use the Cython-generated .cc files rather than the *.pxd/.pyx sources
# used to generate them. Consequently, modifications to the .pyx files will not
# influence the build unless the .cc files are regenerated using Cython.

python_LTLIBRARIES = pywrapfst.la

pyexec_LTILIBRARIES = pywrapfst.la

pywrapfst_la_SOURCES = pywrapfst.cc
pywrapfst_la_CPPFLAGS = -I$(srcdir)/../../include $(PYTHON_CPPFLAGS)
pywrapfst_la_LDFLAGS = $(PYTHON_LDFLAGS) -avoid-version -module
pywrapfst_la_LIBADD = ../far/libfstfarscript.la ../far/libfstfar.la \
                      ../../script/libfstscript.la ../../lib/libfst.la \
                      -lm $(DL_LIBS)

# Exports the *.pxd/*.pxd source files.
EXTRA_DIST = basictypes.pxd fst.pxd ios.pxd memory.pxd pywrapfst.pxd \
             pywrapfst.pyx
