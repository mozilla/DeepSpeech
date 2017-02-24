WARNINGS=-W -Wall -Wstrict-prototypes -Wmissing-prototypes -Waggregate-return \
    -Wcast-align -Wcast-qual -Wnested-externs -Wshadow -Wbad-function-cast \
    -Wwrite-strings

ifeq "$(DATATYPE)" ""
 DATATYPE=float
endif

ifeq  "$(DATATYPE)" "int32_t"
 TYPEFLAGS=-DFIXED_POINT=32
endif

ifeq  "$(DATATYPE)" "int16_t"
 TYPEFLAGS=-DFIXED_POINT=16
endif

ifeq  "$(DATATYPE)" "simd"
 TYPEFLAGS=-DUSE_SIMD=1 -msse
endif

ifeq "$(TYPEFLAGS)"  ""
 TYPEFLAGS=-Dkiss_fft_scalar=$(DATATYPE)
endif

ifneq ("$(KISS_FFT_USE_ALLOCA)","")
	CFLAGS+= -DKISS_FFT_USE_ALLOCA=1
endif 
CFLAGS+= $(CFLAGADD)


FFTUTIL=fft_$(DATATYPE)
FASTFILT=fastconv_$(DATATYPE)
FASTFILTREAL=fastconvr_$(DATATYPE)
PSDPNG=psdpng_$(DATATYPE)
DUMPHDR=dumphdr_$(DATATYPE)

all: $(FFTUTIL) $(FASTFILT) $(FASTFILTREAL) 
# 	$(PSDPNG) 
#	$(DUMPHDR)

#CFLAGS=-Wall -O3 -pedantic -march=pentiumpro -ffast-math -fomit-frame-pointer $(WARNINGS)
# If the above flags do not work, try the following
CFLAGS=-Wall -O3 $(WARNINGS) 
# tip: try -openmp or -fopenmp to use multiple cores

$(FASTFILTREAL): ../kiss_fft.c kiss_fastfir.c kiss_fftr.c
	$(CC) -o $@ $(CFLAGS) -I.. $(TYPEFLAGS) -DREAL_FASTFIR  $+ -DFAST_FILT_UTIL -lm 

$(FASTFILT): ../kiss_fft.c kiss_fastfir.c
	$(CC) -o $@ $(CFLAGS) -I.. $(TYPEFLAGS) $+  -DFAST_FILT_UTIL -lm

$(FFTUTIL): ../kiss_fft.c fftutil.c kiss_fftnd.c kiss_fftr.c kiss_fftndr.c
	$(CC) -o $@ $(CFLAGS) -I.. $(TYPEFLAGS) $+ -lm

$(PSDPNG): ../kiss_fft.c psdpng.c kiss_fftr.c
	$(CC) -o $@ $(CFLAGS) -I.. $(TYPEFLAGS) $+ -lpng -lm

$(DUMPHDR): ../kiss_fft.c dumphdr.c
	$(CC) -o $@ $(CFLAGS) -I.. $(TYPEFLAGS) $+ -lm

clean:
	rm -f *~ fft fft_* fastconv fastconv_* fastconvr fastconvr_* psdpng psdpng_*
