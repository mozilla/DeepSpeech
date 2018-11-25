#include "kiss_fftr.h"
#include "_kiss_fft_guts.h"
#include <sys/times.h>
#include <time.h>
#include <unistd.h>

static double cputime(void)
{
    struct tms t;
    times(&t);
    return (double)(t.tms_utime + t.tms_stime)/  sysconf(_SC_CLK_TCK) ;
}

static
kiss_fft_scalar rand_scalar(void) 
{
#ifdef USE_SIMD
    return _mm_set1_ps(rand()-RAND_MAX/2);
#else
    kiss_fft_scalar s = (kiss_fft_scalar)(rand() -RAND_MAX/2);
    return s/2;
#endif
}

static
double snr_compare( kiss_fft_cpx * vec1,kiss_fft_cpx * vec2, int n)
{
    int k;
    double sigpow=1e-10,noisepow=1e-10,err,snr,scale=0;

#ifdef USE_SIMD
    float *fv1 = (float*)vec1;
    float *fv2 = (float*)vec2;
    for (k=0;k<8*n;++k) {
        sigpow += *fv1 * *fv1;
        err = *fv1 - *fv2;
        noisepow += err*err;
        ++fv1;
        ++fv2;
    }
#else
    for (k=0;k<n;++k) {
        sigpow += (double)vec1[k].r * (double)vec1[k].r + 
                  (double)vec1[k].i * (double)vec1[k].i;
        err = (double)vec1[k].r - (double)vec2[k].r;
        noisepow += err * err;
        err = (double)vec1[k].i - (double)vec2[k].i;
        noisepow += err * err;

        if (vec1[k].r)
            scale +=(double) vec2[k].r / (double)vec1[k].r;
    }
#endif    
    snr = 10*log10( sigpow / noisepow );
    scale /= n;
    if (snr<10) {
        printf( "\npoor snr, try a scaling factor %f\n" , scale );
        exit(1);
    }
    return snr;
}

#ifndef NUMFFTS
#define NUMFFTS 10000
#endif


int main(int argc,char ** argv)
{
    int nfft = 8*3*5;
    double ts,tfft,trfft;
    int i;
    if (argc>1)
        nfft = atoi(argv[1]);
    kiss_fft_cpx cin[nfft];
    kiss_fft_cpx cout[nfft];
    kiss_fft_cpx sout[nfft];
    kiss_fft_cfg  kiss_fft_state;
    kiss_fftr_cfg  kiss_fftr_state;

    kiss_fft_scalar rin[nfft+2];
    kiss_fft_scalar rout[nfft+2];
    kiss_fft_scalar zero;
    memset(&zero,0,sizeof(zero) ); // ugly way of setting short,int,float,double, or __m128 to zero

    srand(time(0));

    for (i=0;i<nfft;++i) {
        rin[i] = rand_scalar();
        cin[i].r = rin[i];
        cin[i].i = zero;
    }

    kiss_fft_state = kiss_fft_alloc(nfft,0,0,0);
    kiss_fftr_state = kiss_fftr_alloc(nfft,0,0,0);
    kiss_fft(kiss_fft_state,cin,cout);
    kiss_fftr(kiss_fftr_state,rin,sout);
    /*
    printf(" results from kiss_fft : (%f,%f), (%f,%f), (%f,%f) ...\n "
            , (float)cout[0].r , (float)cout[0].i
            , (float)cout[1].r , (float)cout[1].i
            , (float)cout[2].r , (float)cout[2].i); 
    printf(" results from kiss_fftr: (%f,%f), (%f,%f), (%f,%f) ...\n "
            , (float)sout[0].r , (float)sout[0].i
            , (float)sout[1].r , (float)sout[1].i
            , (float)sout[2].r , (float)sout[2].i); 
    */
        
    printf( "nfft=%d, inverse=%d, snr=%g\n",
            nfft,0, snr_compare(cout,sout,(nfft/2)+1) );
    ts = cputime();
    for (i=0;i<NUMFFTS;++i) {
        kiss_fft(kiss_fft_state,cin,cout);
    }
    tfft = cputime() - ts;
    
    ts = cputime();
    for (i=0;i<NUMFFTS;++i) {
        kiss_fftr( kiss_fftr_state, rin, cout );
        /* kiss_fftri(kiss_fftr_state,cout,rin); */
    }
    trfft = cputime() - ts;

    printf("%d complex ffts took %gs, real took %gs\n",NUMFFTS,tfft,trfft);

    free(kiss_fft_state);
    free(kiss_fftr_state);

    kiss_fft_state = kiss_fft_alloc(nfft,1,0,0);
    kiss_fftr_state = kiss_fftr_alloc(nfft,1,0,0);

    memset(cin,0,sizeof(cin));
#if 1
    for (i=1;i< nfft/2;++i) {
        //cin[i].r = (kiss_fft_scalar)(rand()-RAND_MAX/2);
        cin[i].r = rand_scalar();
        cin[i].i = rand_scalar();
    }
#else
    cin[0].r = 12000;
    cin[3].r = 12000;
    cin[nfft/2].r = 12000;
#endif

    // conjugate symmetry of real signal 
    for (i=1;i< nfft/2;++i) {
        cin[nfft-i].r = cin[i].r;
        cin[nfft-i].i = - cin[i].i;
    }

    kiss_fft(kiss_fft_state,cin,cout);
    kiss_fftri(kiss_fftr_state,cin,rout);
    /*
    printf(" results from inverse kiss_fft : (%f,%f), (%f,%f), (%f,%f), (%f,%f), (%f,%f) ...\n "
            , (float)cout[0].r , (float)cout[0].i , (float)cout[1].r , (float)cout[1].i , (float)cout[2].r , (float)cout[2].i , (float)cout[3].r , (float)cout[3].i , (float)cout[4].r , (float)cout[4].i
            ); 

    printf(" results from inverse kiss_fftr: %f,%f,%f,%f,%f ... \n"
            ,(float)rout[0] ,(float)rout[1] ,(float)rout[2] ,(float)rout[3] ,(float)rout[4]);
*/
    for (i=0;i<nfft;++i) {
        sout[i].r = rout[i];
        sout[i].i = zero;
    }

    printf( "nfft=%d, inverse=%d, snr=%g\n",
            nfft,1, snr_compare(cout,sout,nfft/2) );
    free(kiss_fft_state);
    free(kiss_fftr_state);

    return 0;
}
