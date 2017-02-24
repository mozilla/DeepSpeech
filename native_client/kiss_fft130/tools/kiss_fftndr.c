/*
Copyright (c) 2003-2004, Mark Borgerding

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the author nor the names of any contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "kiss_fftndr.h"
#include "_kiss_fft_guts.h"
#define MAX(x,y) ( ( (x)<(y) )?(y):(x) )

struct kiss_fftndr_state
{
    int dimReal;
    int dimOther;
    kiss_fftr_cfg cfg_r;
    kiss_fftnd_cfg cfg_nd;
    void * tmpbuf;
};

static int prod(const int *dims, int ndims)
{
    int x=1;
    while (ndims--) 
        x *= *dims++;
    return x;
}

kiss_fftndr_cfg kiss_fftndr_alloc(const int *dims,int ndims,int inverse_fft,void*mem,size_t*lenmem)
{
    kiss_fftndr_cfg st = NULL;
    size_t nr=0 , nd=0,ntmp=0;
    int dimReal = dims[ndims-1];
    int dimOther = prod(dims,ndims-1);
    size_t memneeded;
    
    (void)kiss_fftr_alloc(dimReal,inverse_fft,NULL,&nr);
    (void)kiss_fftnd_alloc(dims,ndims-1,inverse_fft,NULL,&nd);
    ntmp =
        MAX( 2*dimOther , dimReal+2) * sizeof(kiss_fft_scalar)  // freq buffer for one pass
        + dimOther*(dimReal+2) * sizeof(kiss_fft_scalar);  // large enough to hold entire input in case of in-place

    memneeded = sizeof( struct kiss_fftndr_state ) + nr + nd + ntmp;

    if (lenmem==NULL) {
        st = (kiss_fftndr_cfg) malloc(memneeded);
    }else{
        if (*lenmem >= memneeded)
            st = (kiss_fftndr_cfg)mem;
        *lenmem = memneeded; 
    }
    if (st==NULL)
        return NULL;
    memset( st , 0 , memneeded);
    
    st->dimReal = dimReal;
    st->dimOther = dimOther;
    st->cfg_r = kiss_fftr_alloc( dimReal,inverse_fft,st+1,&nr);
    st->cfg_nd = kiss_fftnd_alloc(dims,ndims-1,inverse_fft, ((char*) st->cfg_r)+nr,&nd);
    st->tmpbuf = (char*)st->cfg_nd + nd;

    return st;
}

void kiss_fftndr(kiss_fftndr_cfg st,const kiss_fft_scalar *timedata,kiss_fft_cpx *freqdata)
{
    int k1,k2;
    int dimReal = st->dimReal;
    int dimOther = st->dimOther;
    int nrbins = dimReal/2+1;

    kiss_fft_cpx * tmp1 = (kiss_fft_cpx*)st->tmpbuf; 
    kiss_fft_cpx * tmp2 = tmp1 + MAX(nrbins,dimOther);

    // timedata is N0 x N1 x ... x Nk real

    // take a real chunk of data, fft it and place the output at correct intervals
    for (k1=0;k1<dimOther;++k1) {
        kiss_fftr( st->cfg_r, timedata + k1*dimReal , tmp1 ); // tmp1 now holds nrbins complex points
        for (k2=0;k2<nrbins;++k2)
           tmp2[ k2*dimOther+k1 ] = tmp1[k2];
    }

    for (k2=0;k2<nrbins;++k2) {
        kiss_fftnd(st->cfg_nd, tmp2+k2*dimOther, tmp1);  // tmp1 now holds dimOther complex points
        for (k1=0;k1<dimOther;++k1) 
            freqdata[ k1*(nrbins) + k2] = tmp1[k1];
    }
}

void kiss_fftndri(kiss_fftndr_cfg st,const kiss_fft_cpx *freqdata,kiss_fft_scalar *timedata)
{
    int k1,k2;
    int dimReal = st->dimReal;
    int dimOther = st->dimOther;
    int nrbins = dimReal/2+1;
    kiss_fft_cpx * tmp1 = (kiss_fft_cpx*)st->tmpbuf; 
    kiss_fft_cpx * tmp2 = tmp1 + MAX(nrbins,dimOther);

    for (k2=0;k2<nrbins;++k2) {
        for (k1=0;k1<dimOther;++k1) 
            tmp1[k1] = freqdata[ k1*(nrbins) + k2 ];
        kiss_fftnd(st->cfg_nd, tmp1, tmp2+k2*dimOther);
    }

    for (k1=0;k1<dimOther;++k1) {
        for (k2=0;k2<nrbins;++k2)
            tmp1[k2] = tmp2[ k2*dimOther+k1 ];
        kiss_fftri( st->cfg_r,tmp1,timedata + k1*dimReal);
    }
}
