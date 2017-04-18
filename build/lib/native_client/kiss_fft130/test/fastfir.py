#!/usr/bin/env python

from Numeric import *
from FFT import *

def make_random(len):
    import random
    res=[]
    for i in range(int(len)):
        r=random.uniform(-1,1)
        i=random.uniform(-1,1)
        res.append( complex(r,i) )
    return res

def slowfilter(sig,h):
    translen = len(h)-1
    return convolve(sig,h)[translen:-translen]

def nextpow2(x):
    return 2 ** math.ceil(math.log(x)/math.log(2))

def fastfilter(sig,h,nfft=None):
    if nfft is None:
        nfft = int( nextpow2( 2*len(h) ) )
    H = fft( h , nfft )
    scraplen = len(h)-1
    keeplen = nfft-scraplen
    res=[]
    isdone = 0
    lastidx = nfft
    idx0 = 0
    while not isdone:
        idx1 = idx0 + nfft
        if idx1 >= len(sig):
            idx1 = len(sig)
            lastidx = idx1-idx0
            if lastidx <= scraplen:
                break
            isdone = 1
        Fss = fft(sig[idx0:idx1],nfft)
        fm = Fss * H
        m = inverse_fft(fm)
        res.append( m[scraplen:lastidx] )
        idx0 += keeplen
    return concatenate( res )

def main():
    import sys
    from getopt import getopt
    opts,args = getopt(sys.argv[1:],'rn:l:')
    opts=dict(opts)

    siglen = int(opts.get('-l',1e4 ) )
    hlen =50 
 
    nfft = int(opts.get('-n',128) )
    usereal = opts.has_key('-r')

    print 'nfft=%d'%nfft
    # make a signal
    sig = make_random( siglen )
    # make an impulse response
    h = make_random( hlen )
    #h=[1]*2+[0]*3
    if usereal:
        sig=[c.real for c in sig]
        h=[c.real for c in h]

    # perform MAC filtering
    yslow = slowfilter(sig,h)
    #print '<YSLOW>',yslow,'</YSLOW>'
    #yfast = fastfilter(sig,h,nfft)
    yfast = utilfastfilter(sig,h,nfft,usereal)
    #print yfast
    print 'len(yslow)=%d'%len(yslow)
    print 'len(yfast)=%d'%len(yfast)
    diff = yslow-yfast
    snr = 10*log10( abs( vdot(yslow,yslow) / vdot(diff,diff) ) )
    print 'snr=%s' % snr
    if snr < 10.0:
        print 'h=',h
        print 'sig=',sig[:5],'...'
        print 'yslow=',yslow[:5],'...'
        print 'yfast=',yfast[:5],'...'

def utilfastfilter(sig,h,nfft,usereal):
    import compfft
    import os
    open( 'sig.dat','w').write( compfft.dopack(sig,'f',not usereal) )
    open( 'h.dat','w').write( compfft.dopack(h,'f',not usereal) )
    if usereal: 
        util = './fastconvr' 
    else:
        util = './fastconv'
    cmd = 'time %s -n %d -i sig.dat -h h.dat -o out.dat' % (util, nfft)
    print cmd
    ec  = os.system(cmd)
    print 'exited->',ec
    return compfft.dounpack(open('out.dat').read(),'f',not usereal)

if __name__ == "__main__":
    main()
