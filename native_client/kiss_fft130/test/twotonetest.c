#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "kiss_fft.h"
#include "kiss_fftr.h"
#include <limits.h>


static
double two_tone_test( int nfft, int bin1,int bin2)
{
    kiss_fftr_cfg cfg	= NULL;
    kiss_fft_cpx *kout	= NULL;
    kiss_fft_scalar *tbuf = NULL;

    int i;
    double f1 = bin1*2*M_PI/nfft;
    double f2 = bin2*2*M_PI/nfft;
    double sigpow=0;
    double noisepow=0;
#if FIXED_POINT==32
    long maxrange = LONG_MAX;
#else
    long maxrange = SHRT_MAX;/* works fine for float too*/
#endif

    cfg = kiss_fftr_alloc(nfft , 0, NULL, NULL);
    tbuf	= KISS_FFT_MALLOC(nfft * sizeof(kiss_fft_scalar));
    kout	= KISS_FFT_MALLOC(nfft * sizeof(kiss_fft_cpx));

    /* generate a signal with two tones*/
    for (i = 0; i < nfft; i++) {
#ifdef USE_SIMD
        tbuf[i] = _mm_set1_ps( (maxrange>>1)*cos(f1*i)
                             + (maxrange>>1)*cos(f2*i) );
#else        
        tbuf[i] =  (maxrange>>1)*cos(f1*i)
                 + (maxrange>>1)*cos(f2*i);
#endif        
    }

    kiss_fftr(cfg, tbuf, kout);

    for (i=0;i < (nfft/2+1);++i) {
#ifdef USE_SIMD        
        double tmpr = (double)*(float*)&kout[i].r / (double)maxrange;
        double tmpi = (double)*(float*)&kout[i].i / (double)maxrange;
#else
        double tmpr = (double)kout[i].r / (double)maxrange;
        double tmpi = (double)kout[i].i / (double)maxrange;
#endif
        double mag2 = tmpr*tmpr + tmpi*tmpi;
        if (i!=0 && i!= nfft/2)
            mag2 *= 2; /* all bins except DC and Nyquist have symmetric counterparts implied*/

        /* if there is power in one of the expected bins, it is signal, otherwise noise*/
        if ( i!=bin1 && i != bin2 ) 
            noisepow += mag2;
        else
            sigpow += mag2;
    }
    kiss_fft_cleanup();
    /*printf("TEST %d,%d,%d noise @ %fdB\n",nfft,bin1,bin2,10*log10(noisepow/sigpow +1e-30) );*/
    return 10*log10(sigpow/(noisepow+1e-50) );
}

int main(int argc,char ** argv)
{
    int nfft = 4*2*2*3*5;
    if (argc>1) nfft = atoi(argv[1]);

    int i,j;
    double minsnr = 500;
    double maxsnr = -500;
    double snr;
    for (i=0;i<nfft/2;i+= (nfft>>4)+1) {
        for (j=i;j<nfft/2;j+=(nfft>>4)+7) {
            snr = two_tone_test(nfft,i,j);
            if (snr<minsnr) {
                minsnr=snr;
            }
            if (snr>maxsnr) {
                maxsnr=snr;
            }
        }
    }
    snr = two_tone_test(nfft,nfft/2,nfft/2);
    if (snr<minsnr) minsnr=snr;
    if (snr>maxsnr) maxsnr=snr;

    printf("TwoToneTest: snr ranges from %ddB to %ddB\n",(int)minsnr,(int)maxsnr);
    printf("sizeof(kiss_fft_scalar) = %d\n",(int)sizeof(kiss_fft_scalar) );
    return 0;
}
