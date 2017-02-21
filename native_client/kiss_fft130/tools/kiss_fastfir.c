/*
Copyright (c) 2003-2004, Mark Borgerding

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the author nor the names of any contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "_kiss_fft_guts.h"


/*
 Some definitions that allow real or complex filtering
*/
#ifdef REAL_FASTFIR
#define MIN_FFT_LEN 2048
#include "kiss_fftr.h"
typedef kiss_fft_scalar kffsamp_t;
typedef kiss_fftr_cfg kfcfg_t;
#define FFT_ALLOC kiss_fftr_alloc
#define FFTFWD kiss_fftr
#define FFTINV kiss_fftri
#else
#define MIN_FFT_LEN 1024
typedef kiss_fft_cpx kffsamp_t;
typedef kiss_fft_cfg kfcfg_t;
#define FFT_ALLOC kiss_fft_alloc
#define FFTFWD kiss_fft
#define FFTINV kiss_fft
#endif

typedef struct kiss_fastfir_state *kiss_fastfir_cfg;



kiss_fastfir_cfg kiss_fastfir_alloc(const kffsamp_t * imp_resp,size_t n_imp_resp,
        size_t * nfft,void * mem,size_t*lenmem);

/* see do_file_filter for usage */
size_t kiss_fastfir( kiss_fastfir_cfg cfg, kffsamp_t * inbuf, kffsamp_t * outbuf, size_t n, size_t *offset);



static int verbose=0;


struct kiss_fastfir_state{
    size_t nfft;
    size_t ngood;
    kfcfg_t fftcfg;
    kfcfg_t ifftcfg;
    kiss_fft_cpx * fir_freq_resp;
    kiss_fft_cpx * freqbuf;
    size_t n_freq_bins;
    kffsamp_t * tmpbuf;
};


kiss_fastfir_cfg kiss_fastfir_alloc(
        const kffsamp_t * imp_resp,size_t n_imp_resp,
        size_t *pnfft, /* if <= 0, an appropriate size will be chosen */
        void * mem,size_t*lenmem)
{
    kiss_fastfir_cfg st = NULL;
    size_t len_fftcfg,len_ifftcfg;
    size_t memneeded = sizeof(struct kiss_fastfir_state);
    char * ptr;
    size_t i;
    size_t nfft=0;
    float scale;
    int n_freq_bins;
    if (pnfft)
        nfft=*pnfft;

    if (nfft<=0) {
        /* determine fft size as next power of two at least 2x 
         the impulse response length*/
        i=n_imp_resp-1;
        nfft=2;
        do{
             nfft<<=1;
        }while (i>>=1);
#ifdef MIN_FFT_LEN
        if ( nfft < MIN_FFT_LEN )
            nfft=MIN_FFT_LEN;
#endif        
    }
    if (pnfft)
        *pnfft = nfft;

#ifdef REAL_FASTFIR
    n_freq_bins = nfft/2 + 1;
#else
    n_freq_bins = nfft;
#endif
    /*fftcfg*/
    FFT_ALLOC (nfft, 0, NULL, &len_fftcfg);
    memneeded += len_fftcfg;  
    /*ifftcfg*/
    FFT_ALLOC (nfft, 1, NULL, &len_ifftcfg);
    memneeded += len_ifftcfg;  
    /* tmpbuf */
    memneeded += sizeof(kffsamp_t) * nfft;
    /* fir_freq_resp */
    memneeded += sizeof(kiss_fft_cpx) * n_freq_bins;
    /* freqbuf */
    memneeded += sizeof(kiss_fft_cpx) * n_freq_bins;
    
    if (lenmem == NULL) {
        st = (kiss_fastfir_cfg) malloc (memneeded);
    } else {
        if (*lenmem >= memneeded)
            st = (kiss_fastfir_cfg) mem;
        *lenmem = memneeded;
    }
    if (!st)
        return NULL;

    st->nfft = nfft;
    st->ngood = nfft - n_imp_resp + 1;
    st->n_freq_bins = n_freq_bins;
    ptr=(char*)(st+1);

    st->fftcfg = (kfcfg_t)ptr;
    ptr += len_fftcfg;

    st->ifftcfg = (kfcfg_t)ptr;
    ptr += len_ifftcfg;

    st->tmpbuf = (kffsamp_t*)ptr;
    ptr += sizeof(kffsamp_t) * nfft;

    st->freqbuf = (kiss_fft_cpx*)ptr;
    ptr += sizeof(kiss_fft_cpx) * n_freq_bins;
    
    st->fir_freq_resp = (kiss_fft_cpx*)ptr;
    ptr += sizeof(kiss_fft_cpx) * n_freq_bins;

    FFT_ALLOC (nfft,0,st->fftcfg , &len_fftcfg);
    FFT_ALLOC (nfft,1,st->ifftcfg , &len_ifftcfg);

    memset(st->tmpbuf,0,sizeof(kffsamp_t)*nfft);
    /*zero pad in the middle to left-rotate the impulse response 
      This puts the scrap samples at the end of the inverse fft'd buffer */
    st->tmpbuf[0] = imp_resp[ n_imp_resp - 1 ];
    for (i=0;i<n_imp_resp - 1; ++i) {
        st->tmpbuf[ nfft - n_imp_resp + 1 + i ] = imp_resp[ i ];
    }

    FFTFWD(st->fftcfg,st->tmpbuf,st->fir_freq_resp);

    /* TODO: this won't work for fixed point */
    scale = 1.0 / st->nfft;

    for ( i=0; i < st->n_freq_bins; ++i ) {
#ifdef USE_SIMD
        st->fir_freq_resp[i].r *= _mm_set1_ps(scale);
        st->fir_freq_resp[i].i *= _mm_set1_ps(scale);
#else
        st->fir_freq_resp[i].r *= scale;
        st->fir_freq_resp[i].i *= scale;
#endif
    }
    return st;
}

static void fastconv1buf(const kiss_fastfir_cfg st,const kffsamp_t * in,kffsamp_t * out)
{
    size_t i;
    /* multiply the frequency response of the input signal by
     that of the fir filter*/
    FFTFWD( st->fftcfg, in , st->freqbuf );
    for ( i=0; i<st->n_freq_bins; ++i ) {
        kiss_fft_cpx tmpsamp; 
        C_MUL(tmpsamp,st->freqbuf[i],st->fir_freq_resp[i]);
        st->freqbuf[i] = tmpsamp;
    }

    /* perform the inverse fft*/
    FFTINV(st->ifftcfg,st->freqbuf,out);
}

/* n : the size of inbuf and outbuf in samples
   return value: the number of samples completely processed
   n-retval samples should be copied to the front of the next input buffer */
static size_t kff_nocopy(
        kiss_fastfir_cfg st,
        const kffsamp_t * inbuf, 
        kffsamp_t * outbuf,
        size_t n)
{
    size_t norig=n;
    while (n >= st->nfft ) {
        fastconv1buf(st,inbuf,outbuf);
        inbuf += st->ngood;
        outbuf += st->ngood;
        n -= st->ngood;
    }
    return norig - n;
}

static
size_t kff_flush(kiss_fastfir_cfg st,const kffsamp_t * inbuf,kffsamp_t * outbuf,size_t n)
{
    size_t zpad=0,ntmp;

    ntmp = kff_nocopy(st,inbuf,outbuf,n);
    n -= ntmp;
    inbuf += ntmp;
    outbuf += ntmp;

    zpad = st->nfft - n;
    memset(st->tmpbuf,0,sizeof(kffsamp_t)*st->nfft );
    memcpy(st->tmpbuf,inbuf,sizeof(kffsamp_t)*n );
    
    fastconv1buf(st,st->tmpbuf,st->tmpbuf);
    
    memcpy(outbuf,st->tmpbuf,sizeof(kffsamp_t)*( st->ngood - zpad ));
    return ntmp + st->ngood - zpad;
}

size_t kiss_fastfir(
        kiss_fastfir_cfg vst,
        kffsamp_t * inbuf,
        kffsamp_t * outbuf,
        size_t n_new,
        size_t *offset)
{
    size_t ntot = n_new + *offset;
    if (n_new==0) {
        return kff_flush(vst,inbuf,outbuf,ntot);
    }else{
        size_t nwritten = kff_nocopy(vst,inbuf,outbuf,ntot);
        *offset = ntot - nwritten;
        /*save the unused or underused samples at the front of the input buffer */
        memcpy( inbuf , inbuf+nwritten , *offset * sizeof(kffsamp_t) );
        return nwritten;
    }
}

#ifdef FAST_FILT_UTIL
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <assert.h>

static
void direct_file_filter(
        FILE * fin,
        FILE * fout,
        const kffsamp_t * imp_resp,
        size_t n_imp_resp)
{
    size_t nlag = n_imp_resp - 1;

    const kffsamp_t *tmph;
    kffsamp_t *buf, *circbuf;
    kffsamp_t outval;
    size_t nread;
    size_t nbuf;
    size_t oldestlag = 0;
    size_t k, tap;
#ifndef REAL_FASTFIR
    kffsamp_t tmp;
#endif    

    nbuf = 4096;
    buf = (kffsamp_t *) malloc ( sizeof (kffsamp_t) * nbuf);
    circbuf = (kffsamp_t *) malloc (sizeof (kffsamp_t) * nlag);
    if (!circbuf || !buf) {
        perror("circbuf allocation");
        exit(1);
    }

    if ( fread (circbuf, sizeof (kffsamp_t), nlag, fin) !=  nlag ) {
        perror ("insufficient data to overcome transient");
        exit (1);
    }

    do {
        nread = fread (buf, sizeof (kffsamp_t), nbuf, fin);
        if (nread <= 0)
            break;

        for (k = 0; k < nread; ++k) {
            tmph = imp_resp+nlag;
#ifdef REAL_FASTFIR
# ifdef USE_SIMD
            outval = _mm_set1_ps(0);
#else
            outval = 0;
#endif
            for (tap = oldestlag; tap < nlag; ++tap)
                outval += circbuf[tap] * *tmph--;
            for (tap = 0; tap < oldestlag; ++tap)
                outval += circbuf[tap] * *tmph--;
            outval += buf[k] * *tmph;
#else
# ifdef USE_SIMD
            outval.r = outval.i = _mm_set1_ps(0);
#else            
            outval.r = outval.i = 0;
#endif            
            for (tap = oldestlag; tap < nlag; ++tap){
                C_MUL(tmp,circbuf[tap],*tmph);
                --tmph;
                C_ADDTO(outval,tmp);
            }
            
            for (tap = 0; tap < oldestlag; ++tap) {
                C_MUL(tmp,circbuf[tap],*tmph);
                --tmph;
                C_ADDTO(outval,tmp);
            }
            C_MUL(tmp,buf[k],*tmph);
            C_ADDTO(outval,tmp);
#endif

            circbuf[oldestlag++] = buf[k];
            buf[k] = outval;

            if (oldestlag == nlag)
                oldestlag = 0;
        }

        if (fwrite (buf, sizeof (buf[0]), nread, fout) != nread) {
            perror ("short write");
            exit (1);
        }
    } while (nread);
    free (buf);
    free (circbuf);
}

static
void do_file_filter(
        FILE * fin,
        FILE * fout,
        const kffsamp_t * imp_resp,
        size_t n_imp_resp,
        size_t nfft )
{
    int fdout;
    size_t n_samps_buf;

    kiss_fastfir_cfg cfg;
    kffsamp_t *inbuf,*outbuf;
    int nread,nwrite;
    size_t idx_inbuf;

    fdout = fileno(fout);

    cfg=kiss_fastfir_alloc(imp_resp,n_imp_resp,&nfft,0,0);

    /* use length to minimize buffer shift*/
    n_samps_buf = 8*4096/sizeof(kffsamp_t); 
    n_samps_buf = nfft + 4*(nfft-n_imp_resp+1);

    if (verbose) fprintf(stderr,"bufsize=%d\n",(int)(sizeof(kffsamp_t)*n_samps_buf) );
     

    /*allocate space and initialize pointers */
    inbuf = (kffsamp_t*)malloc(sizeof(kffsamp_t)*n_samps_buf);
    outbuf = (kffsamp_t*)malloc(sizeof(kffsamp_t)*n_samps_buf);

    idx_inbuf=0;
    do{
        /* start reading at inbuf[idx_inbuf] */
        nread = fread( inbuf + idx_inbuf, sizeof(kffsamp_t), n_samps_buf - idx_inbuf,fin );

        /* If nread==0, then this is a flush.
            The total number of samples in input is idx_inbuf + nread . */
        nwrite = kiss_fastfir(cfg, inbuf, outbuf,nread,&idx_inbuf) * sizeof(kffsamp_t);
        /* kiss_fastfir moved any unused samples to the front of inbuf and updated idx_inbuf */

        if ( write(fdout, outbuf, nwrite) != nwrite ) {
            perror("short write");
            exit(1);
        }
    }while ( nread );
    free(cfg);
    free(inbuf);
    free(outbuf);
}

int main(int argc,char**argv)
{
    kffsamp_t * h;
    int use_direct=0;
    size_t nh,nfft=0;
    FILE *fin=stdin;
    FILE *fout=stdout;
    FILE *filtfile=NULL;
    while (1) {
        int c=getopt(argc,argv,"n:h:i:o:vd");
        if (c==-1) break;
        switch (c) {
            case 'v':
                verbose=1;
                break;
            case 'n':
                nfft=atoi(optarg);
                break;
            case 'i':
                fin = fopen(optarg,"rb");
                if (fin==NULL) {
                    perror(optarg);
                    exit(1);
                }
                break;
            case 'o':
                fout = fopen(optarg,"w+b");
                if (fout==NULL) {
                    perror(optarg);
                    exit(1);
                }
                break;
            case 'h':
                filtfile = fopen(optarg,"rb");
                if (filtfile==NULL) {
                    perror(optarg);
                    exit(1);
                }
                break;
            case 'd':
                use_direct=1;
                break;
            case '?':
                     fprintf(stderr,"usage options:\n"
                            "\t-n nfft: fft size to use\n"
                            "\t-d : use direct FIR filtering, not fast convolution\n"
                            "\t-i filename: input file\n"
                            "\t-o filename: output(filtered) file\n"
                            "\t-n nfft: fft size to use\n"
                            "\t-h filename: impulse response\n");
                     exit (1);
            default:fprintf(stderr,"bad %c\n",c);break;
        }
    }
    if (filtfile==NULL) {
        fprintf(stderr,"You must supply the FIR coeffs via -h\n");
        exit(1);
    }
    fseek(filtfile,0,SEEK_END);
    nh = ftell(filtfile) / sizeof(kffsamp_t);
    if (verbose) fprintf(stderr,"%d samples in FIR filter\n",(int)nh);
    h = (kffsamp_t*)malloc(sizeof(kffsamp_t)*nh);
    fseek(filtfile,0,SEEK_SET);
    if (fread(h,sizeof(kffsamp_t),nh,filtfile) != nh)
        fprintf(stderr,"short read on filter file\n");

    fclose(filtfile);
 
    if (use_direct)
        direct_file_filter( fin, fout, h,nh);
    else
        do_file_filter( fin, fout, h,nh,nfft);

    if (fout!=stdout) fclose(fout);
    if (fin!=stdin) fclose(fin);

    return 0;
}
#endif
