
WARNINGS=-W -Wall -Wstrict-prototypes -Wmissing-prototypes -Waggregate-return \
    -Wcast-align -Wcast-qual -Wnested-externs -Wshadow -Wbad-function-cast \
    -Wwrite-strings

CFLAGS=-O3 -I.. -I../tools $(WARNINGS)
CFLAGS+=-ffast-math -fomit-frame-pointer 
#CFLAGS+=-funroll-loops
#CFLAGS+=-march=prescott 
#CFLAGS+= -mtune=native 
# TIP: try adding -openmp or -fopenmp  to enable OPENMP directives and use of multiple cores
#CFLAGS+=-fopenmp
CFLAGS+= $(CFLAGADD)


ifeq "$(NFFT)" ""
 NFFT=1800
endif
ifeq "$(NUMFFTS)" ""
 NUMFFTS=10000
endif

ifeq "$(DATATYPE)" ""
 DATATYPE=float
endif

BENCHKISS=bm_kiss_$(DATATYPE)
BENCHFFTW=bm_fftw_$(DATATYPE)
SELFTEST=st_$(DATATYPE)
TESTREAL=tr_$(DATATYPE)
TESTKFC=tkfc_$(DATATYPE)
FASTFILTREAL=ffr_$(DATATYPE)
SELFTESTSRC=twotonetest.c


TYPEFLAGS=-Dkiss_fft_scalar=$(DATATYPE)

ifeq  "$(DATATYPE)" "int16_t" 
 TYPEFLAGS=-DFIXED_POINT=16
endif

ifeq  "$(DATATYPE)" "int32_t"
 TYPEFLAGS=-DFIXED_POINT=32 
endif

ifeq  "$(DATATYPE)" "simd"
 TYPEFLAGS=-DUSE_SIMD=1 -msse
endif


ifeq  "$(DATATYPE)" "float"
 # fftw needs to be built with --enable-float to build this lib
 FFTWLIB=-lfftw3f
else
 FFTWLIB=-lfftw3
endif

FFTWLIBDIR=-L/usr/local/lib/

SRCFILES=../kiss_fft.c ../tools/kiss_fftnd.c ../tools/kiss_fftr.c pstats.c ../tools/kfc.c ../tools/kiss_fftndr.c

all: tools $(BENCHKISS) $(SELFTEST) $(BENCHFFTW) $(TESTREAL) $(TESTKFC)

tools:
	cd ../tools && make all


$(SELFTEST): $(SELFTESTSRC) $(SRCFILES)
	$(CC) -o $@ $(CFLAGS) $(TYPEFLAGS) $+ -lm 

$(TESTKFC): $(SRCFILES) 
	$(CC) -o $@ $(CFLAGS)  -DKFC_TEST $(TYPEFLAGS) $+ -lm
	
$(TESTREAL): test_real.c $(SRCFILES)
	$(CC) -o $@ $(CFLAGS) $(TYPEFLAGS) $+ -lm

$(BENCHKISS): benchkiss.c $(SRCFILES)
	$(CC) -o $@ $(CFLAGS) $(TYPEFLAGS)  $+ -lm

$(BENCHFFTW): benchfftw.c pstats.c
	@echo "======attempting to build FFTW benchmark"
	@$(CC) -o $@ $(CFLAGS) -DDATATYPE$(DATATYPE) $+ $(FFTWLIB) $(FFTWLIBDIR) -lm || echo "FFTW not available for comparison"

test: all
	@./$(TESTKFC)
	@echo "======1d & 2-d complex fft self test (type= $(DATATYPE) )"
	@./$(SELFTEST)
	@echo "======real FFT (type= $(DATATYPE) )"
	@./$(TESTREAL)
	@echo "======timing test (type=$(DATATYPE))"
	@./$(BENCHKISS) -x $(NUMFFTS) -n $(NFFT) 
	@[ -x ./$(BENCHFFTW) ] && ./$(BENCHFFTW) -x $(NUMFFTS) -n $(NFFT) ||true
	@echo "======higher dimensions type=$(DATATYPE))"
	@./testkiss.py

selftest.c:
	./mk_test.py 10 12 14 > selftest.c
selftest_short.c:
	./mk_test.py -s 10 12 14 > selftest_short.c


CXXFLAGS=-O3 -ffast-math -fomit-frame-pointer  -I.. -I../tools -W -Wall
testcpp: testcpp.cc ../kissfft.hh
	$(CXX) -o $@ $(CXXFLAGS) testcpp.cc -lm


clean:
	rm -f *~ bm_* st_* tr_* kf_* tkfc_* ff_* ffr_* *.pyc *.pyo *.dat testcpp
