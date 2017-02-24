#include "kiss_fft.h"


void check(kiss_fft_cpx  * in,kiss_fft_cpx  * out,int nfft,int isinverse)
{
    int bin,k;
    double errpow=0,sigpow=0;
    
    for (bin=0;bin<nfft;++bin) {
        double ansr = 0;
        double ansi = 0;
        double difr;
        double difi;

        for (k=0;k<nfft;++k) {
            double phase = -2*M_PI*bin*k/nfft;
            double re = cos(phase);
            double im = sin(phase);
            if (isinverse)
                im = -im;

#ifdef FIXED_POINT
            re /= nfft;
            im /= nfft;
#endif            

            ansr += in[k].r * re - in[k].i * im;
            ansi += in[k].r * im + in[k].i * re;
        }
        difr = ansr - out[bin].r;
        difi = ansi - out[bin].i;
        errpow += difr*difr + difi*difi;
        sigpow += ansr*ansr+ansi*ansi;
    }
    printf("nfft=%d inverse=%d,snr = %f\n",nfft,isinverse,10*log10(sigpow/errpow) );
}

void test1d(int nfft,int isinverse)
{
    size_t buflen = sizeof(kiss_fft_cpx)*nfft;

    kiss_fft_cpx  * in = (kiss_fft_cpx*)malloc(buflen);
    kiss_fft_cpx  * out= (kiss_fft_cpx*)malloc(buflen);
    kiss_fft_cfg  cfg = kiss_fft_alloc(nfft,isinverse,0,0);
    int k;

    for (k=0;k<nfft;++k) {
        in[k].r = (rand() % 65536) - 32768;
        in[k].i = (rand() % 65536) - 32768;
    }

    kiss_fft(cfg,in,out);

    check(in,out,nfft,isinverse);

    free(in);
    free(out);
    free(cfg);
}

int main(int argc,char ** argv)
{
    if (argc>1) {
        int k;
        for (k=1;k<argc;++k) {
            test1d(atoi(argv[k]),0);
            test1d(atoi(argv[k]),1);
        }
    }else{
        test1d(32,0);
        test1d(32,1);
    }
    return 0;
}
