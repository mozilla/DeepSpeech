/* this program is in the public domain 
   A program that helps the authors of the fine fftw library benchmark kiss 
*/

#include "bench-user.h"
#include <math.h>

#include "kiss_fft.h"
#include "kiss_fftnd.h"
#include "kiss_fftr.h"

BEGIN_BENCH_DOC
BENCH_DOC("name", "kissfft")
BENCH_DOC("version", "1.0.1")
BENCH_DOC("year", "2004")
BENCH_DOC("author", "Mark Borgerding")
BENCH_DOC("language", "C")
BENCH_DOC("url", "http://sourceforge.net/projects/kissfft/")
BENCH_DOC("copyright",
"Copyright (c) 2003,4 Mark Borgerding\n"
"\n"
"All rights reserved.\n"
"\n"
"Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:\n"
"\n"
"    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.\n"
"    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.\n"
"    * Neither the author nor the names of any contributors may be used to endorse or promote products derived from this software without specific prior written permission.\n"
"\n"
	  "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.\n")
END_BENCH_DOC

int can_do(struct problem *p)
{
    if (p->rank == 1) {
        if (p->kind == PROBLEM_REAL) {
            return (p->n[0] & 1) == 0;  /* only even real is okay */
        } else {
            return 1;
        }
    } else {
        return p->kind == PROBLEM_COMPLEX;
    }
}

static kiss_fft_cfg cfg=NULL;
static kiss_fftr_cfg cfgr=NULL;
static kiss_fftnd_cfg cfgnd=NULL;

#define FAILIF( c ) \
    if (  c ) do {\
    fprintf(stderr,\
    "kissfft: " #c " (file=%s:%d errno=%d %s)\n",\
        __FILE__,__LINE__ , errno,strerror( errno ) ) ;\
    exit(1);\
    }while(0)



void setup(struct problem *p)
{
    size_t i;

    /*
    fprintf(stderr,"%s %s %d-d ",
            (p->sign == 1)?"Inverse":"Forward",
            p->kind == PROBLEM_COMPLEX?"Complex":"Real",
            p->rank);
    */
    if (p->rank == 1) {
        if (p->kind == PROBLEM_COMPLEX) {
            cfg = kiss_fft_alloc (p->n[0], (p->sign == 1), 0, 0);
            FAILIF(cfg==NULL);
        }else{
            cfgr = kiss_fftr_alloc (p->n[0], (p->sign == 1), 0, 0);
            FAILIF(cfgr==NULL);
        }
    }else{
        int dims[5];
        for (i=0;i<p->rank;++i){
            dims[i] = p->n[i];
        }
        /* multi-dimensional */
        if (p->kind == PROBLEM_COMPLEX) {
            cfgnd = kiss_fftnd_alloc( dims , p->rank, (p->sign == 1), 0, 0 );
            FAILIF(cfgnd==NULL);
        }
    }
}

void doit(int iter, struct problem *p)
{
    int i;
    void *in = p->in;
    void *out = p->out;

    if (p->in_place)
        out = p->in;

    if (p->rank == 1) {
        if (p->kind == PROBLEM_COMPLEX){
            for (i = 0; i < iter; ++i)
                kiss_fft (cfg, in, out);
        } else {
            /* PROBLEM_REAL */
            if (p->sign == -1)   /* FORWARD */
                for (i = 0; i < iter; ++i)
                    kiss_fftr (cfgr, in, out);
            else
                for (i = 0; i < iter; ++i)
                    kiss_fftri (cfgr, in, out);
        }
    }else{
        /* multi-dimensional */
        for (i = 0; i < iter; ++i)
            kiss_fftnd(cfgnd,in,out);
    }
}

void done(struct problem *p)
{
     free(cfg);
     cfg=NULL;
     free(cfgr);
     cfgr=NULL;
     free(cfgnd);
     cfgnd=NULL;
     UNUSED(p);
}
