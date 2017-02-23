#!/usr/bin/env python

import FFT
import sys
import random
import re
j=complex(0,1)

def randvec(n,iscomplex):
    if iscomplex:
        return [
                int(random.uniform(-32768,32767) ) + j*int(random.uniform(-32768,32767) )
                for i in range(n) ]
    else:                
        return [ int(random.uniform(-32768,32767) ) for i in range(n) ]
    
def c_format(v,round=0):
    if round:
        return ','.join( [ '{%d,%d}' %(int(c.real),int(c.imag) ) for c in v ] ) 
    else:
        s= ','.join( [ '{%.60f ,%.60f }' %(c.real,c.imag) for c in v ] ) 
        return re.sub(r'\.?0+ ',' ',s)

def test_cpx( n,inverse ,short):
    v = randvec(n,1)
    scale = 1
    if short:
        minsnr=30
    else:
        minsnr=100

    if inverse:
        tvecout = FFT.inverse_fft(v)
        if short:
            scale = 1
        else:            
            scale = len(v)
    else:
        tvecout = FFT.fft(v)
        if short:
            scale = 1.0/len(v)

    tvecout = [ c * scale for c in tvecout ]


    s="""#define NFFT %d""" % len(v) + """
    {
        double snr;
        kiss_fft_cpx test_vec_in[NFFT] = { """  + c_format(v) + """};
        kiss_fft_cpx test_vec_out[NFFT] = {"""  + c_format( tvecout ) + """};
        kiss_fft_cpx testbuf[NFFT];
        void * cfg = kiss_fft_alloc(NFFT,%d,0,0);""" % inverse + """

        kiss_fft(cfg,test_vec_in,testbuf);
        snr = snr_compare(test_vec_out,testbuf,NFFT);
        printf("DATATYPE=" xstr(kiss_fft_scalar) ", FFT n=%d, inverse=%d, snr = %g dB\\n",NFFT,""" + str(inverse) + """,snr);
        if (snr<""" + str(minsnr) + """)
            exit_code++;
        free(cfg);
    }
#undef NFFT    
"""
    return s

def compare_func():
    s="""
#define xstr(s) str(s)
#define str(s) #s
double snr_compare( kiss_fft_cpx * test_vec_out,kiss_fft_cpx * testbuf, int n)
{
    int k;
    double sigpow,noisepow,err,snr,scale=0;
    kiss_fft_cpx err;
    sigpow = noisepow = .000000000000000000000000000001; 

    for (k=0;k<n;++k) {
        sigpow += test_vec_out[k].r * test_vec_out[k].r + 
                  test_vec_out[k].i * test_vec_out[k].i;
        C_SUB(err,test_vec_out[k],testbuf[k].r);
        noisepow += err.r * err.r + err.i + err.i;

        if (test_vec_out[k].r)
            scale += testbuf[k].r / test_vec_out[k].r;
    }
    snr = 10*log10( sigpow / noisepow );
    scale /= n;
    if (snr<10)
        printf( "\\npoor snr, try a scaling factor %f\\n" , scale );
    return snr;
}
"""
    return s

def main():

    from getopt import getopt
    opts,args = getopt(sys.argv[1:],'s')
    opts = dict(opts)
    short = int( opts.has_key('-s') )

    fftsizes = args
    if not fftsizes:
        fftsizes = [ 1800 ]
    print '#include "kiss_fft.h"'
    print compare_func()
    print "int main() { int exit_code=0;\n"
    for n in fftsizes:
        n = int(n)
        print test_cpx(n,0,short)
        print test_cpx(n,1,short)
    print """
    return exit_code;
}
"""

if __name__ == "__main__":
    main()
