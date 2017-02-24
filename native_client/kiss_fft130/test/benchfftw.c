#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <unistd.h>
#include "pstats.h"

#ifdef DATATYPEdouble

#define CPXTYPE fftw_complex
#define PLAN fftw_plan
#define FFTMALLOC fftw_malloc
#define MAKEPLAN fftw_plan_dft_1d
#define DOFFT fftw_execute
#define DESTROYPLAN fftw_destroy_plan
#define FFTFREE fftw_free

#elif defined(DATATYPEfloat)

#define CPXTYPE fftwf_complex
#define PLAN fftwf_plan
#define FFTMALLOC fftwf_malloc
#define MAKEPLAN fftwf_plan_dft_1d
#define DOFFT fftwf_execute
#define DESTROYPLAN fftwf_destroy_plan
#define FFTFREE fftwf_free

#endif

#ifndef CPXTYPE
int main(void)
{
    fprintf(stderr,"Datatype not available in FFTW\n" );
    return 0;
}
#else
int main(int argc,char ** argv)
{
    int nfft=1024;
    int isinverse=0;
    int numffts=1000,i;

    CPXTYPE * in=NULL;
    CPXTYPE * out=NULL;
    PLAN p;

    pstats_init();

    while (1) {
      int c = getopt (argc, argv, "n:ix:h");
      if (c == -1)
        break;
      switch (c) {
      case 'n':
        nfft = atoi (optarg);
        break;
      case 'x':
        numffts = atoi (optarg);
        break;
      case 'i':
        isinverse = 1;
        break;
      case 'h':
      case '?':
      default:
        fprintf(stderr,"options:\n-n N: complex fft length\n-i: inverse\n-x N: number of ffts to compute\n"
                "");
      }
    }

    in=FFTMALLOC(sizeof(CPXTYPE) * nfft);
    out=FFTMALLOC(sizeof(CPXTYPE) * nfft);
    for (i=0;i<nfft;++i ) {
        in[i][0] = rand() - RAND_MAX/2;
        in[i][1] = rand() - RAND_MAX/2;
    }

    if ( isinverse )
        p = MAKEPLAN(nfft, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    else    
        p = MAKEPLAN(nfft, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    for (i=0;i<numffts;++i)
        DOFFT(p);

    DESTROYPLAN(p);

    FFTFREE(in); FFTFREE(out);

    fprintf(stderr,"fftw\tnfft=%d\tnumffts=%d\n", nfft,numffts);
    pstats_report();

    return 0;
}
#endif
