/*
Copyright (c) 2003-2004, Mark Borgerding

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the author nor the names of any contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "kiss_fft.h"
#include "kiss_fftndr.h"

static
void fft_file(FILE * fin,FILE * fout,int nfft,int isinverse)
{
    kiss_fft_cfg st;
    kiss_fft_cpx * buf;
    kiss_fft_cpx * bufout;

    buf = (kiss_fft_cpx*)malloc(sizeof(kiss_fft_cpx) * nfft );
    bufout = (kiss_fft_cpx*)malloc(sizeof(kiss_fft_cpx) * nfft );
    st = kiss_fft_alloc( nfft ,isinverse ,0,0);

    while ( fread( buf , sizeof(kiss_fft_cpx) * nfft ,1, fin ) > 0 ) {
        kiss_fft( st , buf ,bufout);
        fwrite( bufout , sizeof(kiss_fft_cpx) , nfft , fout );
    }
    free(st);
    free(buf);
    free(bufout);
}

static
void fft_filend(FILE * fin,FILE * fout,int *dims,int ndims,int isinverse)
{
    kiss_fftnd_cfg st;
    kiss_fft_cpx *buf;
    int dimprod=1,i;
    for (i=0;i<ndims;++i) 
        dimprod *= dims[i];

    buf = (kiss_fft_cpx *) malloc (sizeof (kiss_fft_cpx) * dimprod);
    st = kiss_fftnd_alloc (dims, ndims, isinverse, 0, 0);

    while (fread (buf, sizeof (kiss_fft_cpx) * dimprod, 1, fin) > 0) {
        kiss_fftnd (st, buf, buf);
        fwrite (buf, sizeof (kiss_fft_cpx), dimprod, fout);
    }
    free (st);
    free (buf);
}



static
void fft_filend_real(FILE * fin,FILE * fout,int *dims,int ndims,int isinverse)
{
    int dimprod=1,i;
    kiss_fftndr_cfg st;
    void *ibuf;
    void *obuf;
    int insize,outsize; // size in bytes

    for (i=0;i<ndims;++i) 
        dimprod *= dims[i];
    insize = outsize = dimprod;
    int rdim = dims[ndims-1];

    if (isinverse)
        insize = insize*2*(rdim/2+1)/rdim;
    else
        outsize = outsize*2*(rdim/2+1)/rdim;

    ibuf = malloc(insize*sizeof(kiss_fft_scalar));
    obuf = malloc(outsize*sizeof(kiss_fft_scalar));

    st = kiss_fftndr_alloc(dims, ndims, isinverse, 0, 0);

    while ( fread (ibuf, sizeof(kiss_fft_scalar), insize,  fin) > 0) {
        if (isinverse) {
            kiss_fftndri(st,
                    (kiss_fft_cpx*)ibuf,
                    (kiss_fft_scalar*)obuf);
        }else{
            kiss_fftndr(st,
                    (kiss_fft_scalar*)ibuf,
                    (kiss_fft_cpx*)obuf);
        }
        fwrite (obuf, sizeof(kiss_fft_scalar), outsize,fout);
    }
    free(st);
    free(ibuf);
    free(obuf);
}

static
void fft_file_real(FILE * fin,FILE * fout,int nfft,int isinverse)
{
    kiss_fftr_cfg st;
    kiss_fft_scalar * rbuf;
    kiss_fft_cpx * cbuf;

    rbuf = (kiss_fft_scalar*)malloc(sizeof(kiss_fft_scalar) * nfft );
    cbuf = (kiss_fft_cpx*)malloc(sizeof(kiss_fft_cpx) * (nfft/2+1) );
    st = kiss_fftr_alloc( nfft ,isinverse ,0,0);

    if (isinverse==0) {
        while ( fread( rbuf , sizeof(kiss_fft_scalar) * nfft ,1, fin ) > 0 ) {
            kiss_fftr( st , rbuf ,cbuf);
            fwrite( cbuf , sizeof(kiss_fft_cpx) , (nfft/2 + 1) , fout );
        }
    }else{
        while ( fread( cbuf , sizeof(kiss_fft_cpx) * (nfft/2+1) ,1, fin ) > 0 ) {
            kiss_fftri( st , cbuf ,rbuf);
            fwrite( rbuf , sizeof(kiss_fft_scalar) , nfft , fout );
        }
    }
    free(st);
    free(rbuf);
    free(cbuf);
}

static
int get_dims(char * arg,int * dims)
{
    char *p0;
    int ndims=0;

    do{
        p0 = strchr(arg,',');
        if (p0)
            *p0++ = '\0';
        dims[ndims++] = atoi(arg);
//         fprintf(stderr,"dims[%d] = %d\n",ndims-1,dims[ndims-1]); 
        arg = p0;
    }while (p0);
    return ndims;
}

int main(int argc,char ** argv)
{
    int isinverse=0;
    int isreal=0;
    FILE *fin=stdin;
    FILE *fout=stdout;
    int ndims=1;
    int dims[32];
    dims[0] = 1024; /*default fft size*/

    while (1) {
        int c=getopt(argc,argv,"n:iR");
        if (c==-1) break;
        switch (c) {
            case 'n':
                ndims = get_dims(optarg,dims);
                break;
            case 'i':isinverse=1;break;
            case 'R':isreal=1;break;
            case '?':
                     fprintf(stderr,"usage options:\n"
                            "\t-n d1[,d2,d3...]: fft dimension(s)\n"
                            "\t-i : inverse\n"
                            "\t-R : real input samples, not complex\n");
                     exit (1);
            default:fprintf(stderr,"bad %c\n",c);break;
        }
    }

    if ( optind < argc ) {
        if (strcmp("-",argv[optind]) !=0)
            fin = fopen(argv[optind],"rb");
        ++optind;
    }

    if ( optind < argc ) {
        if ( strcmp("-",argv[optind]) !=0 ) 
            fout = fopen(argv[optind],"wb");
        ++optind;
    }

    if (ndims==1) {
        if (isreal)
            fft_file_real(fin,fout,dims[0],isinverse);
        else
            fft_file(fin,fout,dims[0],isinverse);
    }else{
        if (isreal)
            fft_filend_real(fin,fout,dims,ndims,isinverse);
        else
            fft_filend(fin,fout,dims,ndims,isinverse);
    }

    if (fout!=stdout) fclose(fout);
    if (fin!=stdin) fclose(fin);

    return 0;
}
