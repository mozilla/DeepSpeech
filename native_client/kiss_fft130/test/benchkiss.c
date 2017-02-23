#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include <unistd.h>
#include "kiss_fft.h"
#include "kiss_fftr.h"
#include "kiss_fftnd.h"
#include "kiss_fftndr.h"

#include "pstats.h"

static
int getdims(int * dims, char * arg)
{
    char *s;
    int ndims=0;
    while ( (s=strtok( arg , ",") ) ) {
        dims[ndims++] = atoi(s);
        //printf("%s=%d\n",s,dims[ndims-1]);
        arg=NULL;
    }
    return ndims;
}

int main(int argc,char ** argv)
{
    int k;
    int nfft[32];
    int ndims = 1;
    int isinverse=0;
    int numffts=1000,i;
    kiss_fft_cpx * buf;
    kiss_fft_cpx * bufout;
    int real = 0;

    nfft[0] = 1024;// default

    while (1) {
        int c = getopt (argc, argv, "n:ix:r");
        if (c == -1)
            break;
        switch (c) {
            case 'r':
                real = 1;
                break;
            case 'n':
                ndims = getdims(nfft, optarg );
                if (nfft[0] != kiss_fft_next_fast_size(nfft[0]) ) {
                    int ng = kiss_fft_next_fast_size(nfft[0]);
                    fprintf(stderr,"warning: %d might be a better choice for speed than %d\n",ng,nfft[0]);
                }
                break;
            case 'x':
                numffts = atoi (optarg);
                break;
            case 'i':
                isinverse = 1;
                break;
        }
    }
    int nbytes = sizeof(kiss_fft_cpx);
    for (k=0;k<ndims;++k)
        nbytes *= nfft[k];

#ifdef USE_SIMD        
    numffts /= 4;
    fprintf(stderr,"since SIMD implementation does 4 ffts at a time, numffts is being reduced to %d\n",numffts);
#endif

    buf=(kiss_fft_cpx*)KISS_FFT_MALLOC(nbytes);
    bufout=(kiss_fft_cpx*)KISS_FFT_MALLOC(nbytes);
    memset(buf,0,nbytes);

    pstats_init();

    if (ndims==1) {
        if (real) {
            kiss_fftr_cfg st = kiss_fftr_alloc( nfft[0] ,isinverse ,0,0);
            if (isinverse)
                for (i=0;i<numffts;++i)
                    kiss_fftri( st ,(kiss_fft_cpx*)buf,(kiss_fft_scalar*)bufout );
            else
                for (i=0;i<numffts;++i)
                    kiss_fftr( st ,(kiss_fft_scalar*)buf,(kiss_fft_cpx*)bufout );
            free(st);
        }else{
            kiss_fft_cfg st = kiss_fft_alloc( nfft[0] ,isinverse ,0,0);
            for (i=0;i<numffts;++i)
                kiss_fft( st ,buf,bufout );
            free(st);
        }
    }else{
        if (real) {
            kiss_fftndr_cfg st = kiss_fftndr_alloc( nfft,ndims ,isinverse ,0,0);
            if (isinverse)
                for (i=0;i<numffts;++i)
                    kiss_fftndri( st ,(kiss_fft_cpx*)buf,(kiss_fft_scalar*)bufout );
            else
                for (i=0;i<numffts;++i)
                    kiss_fftndr( st ,(kiss_fft_scalar*)buf,(kiss_fft_cpx*)bufout );
            free(st);
        }else{
            kiss_fftnd_cfg st= kiss_fftnd_alloc(nfft,ndims,isinverse ,0,0);
            for (i=0;i<numffts;++i)
                kiss_fftnd( st ,buf,bufout );
            free(st);
        }
    }

    free(buf); free(bufout);

    fprintf(stderr,"KISS\tnfft=");
    for (k=0;k<ndims;++k)
        fprintf(stderr, "%d,",nfft[k]);
    fprintf(stderr,"\tnumffts=%d\n" ,numffts);
    pstats_report();

    kiss_fft_cleanup();

    return 0;
}

