#!/usr/bin/env python

# use FFTPACK as a baseline
import FFT
from Numeric import *
import math
import random
import sys
import struct
import fft

pi=math.pi
e=math.e
j=complex(0,1)
lims=(-32768,32767)

def randbuf(n,cpx=1):
    res = array( [ random.uniform( lims[0],lims[1] ) for i in range(n) ] )
    if cpx:
        res = res + j*randbuf(n,0)
    return res

def main():
    from getopt import getopt
    import popen2
    opts,args = getopt( sys.argv[1:],'u:n:Rt:' )
    opts=dict(opts)
    exitcode=0

    util = opts.get('-u','./kf_float')

    try:
        dims = [ int(d) for d in opts['-n'].split(',')]
        cpx = opts.get('-R') is None
        fmt=opts.get('-t','f')
    except KeyError:
        sys.stderr.write("""
        usage: compfft.py 
        -n d1[,d2,d3...]  : FFT dimension(s)
        -u utilname : see sample_code/fftutil.c, default = ./kf_float
        -R : real-optimized version\n""")
        sys.exit(1)

    x = fft.make_random( dims )

    cmd = '%s -n %s ' % ( util, ','.join([ str(d) for d in dims]) )
    if cpx:
        xout = FFT.fftnd(x)
        xout = reshape(xout,(size(xout),))
    else:
        cmd += '-R '
        xout = FFT.real_fft(x)

    proc = popen2.Popen3( cmd , bufsize=len(x) )

    proc.tochild.write( dopack( x , fmt ,cpx ) )
    proc.tochild.close()
    xoutcomp = dounpack( proc.fromchild.read( ) , fmt ,1 )
    #xoutcomp = reshape( xoutcomp , dims )

    sig = xout * conjugate(xout)
    sigpow = sum( sig )

    diff = xout-xoutcomp
    noisepow = sum( diff * conjugate(diff) )

    snr = 10 * math.log10(abs( sigpow / noisepow ) )
    if snr<100:
        print xout
        print xoutcomp
        exitcode=1
    print 'NFFT=%s,SNR = %f dB' % (str(dims),snr)
    sys.exit(exitcode)

def dopack(x,fmt,cpx):
    x = reshape( x, ( size(x),) )
    if cpx:
        s = ''.join( [ struct.pack('ff',c.real,c.imag) for c in x ] )
    else:
        s = ''.join( [ struct.pack('f',c) for c in x ] )
    return s 

def dounpack(x,fmt,cpx):
    uf = fmt * ( len(x) / 4 )
    s = struct.unpack(uf,x)
    if cpx:
        return array(s[::2]) + array( s[1::2] )*j
    else:    
        return array(s )

if __name__ == "__main__":
    main()
