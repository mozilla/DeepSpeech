#!/usr/bin/env python

import math
import sys
import random

pi=math.pi
e=math.e
j=complex(0,1)

def fft(f,inv):
    n=len(f)
    if n==1:
        return f

    for p in 2,3,5:
        if n%p==0:
            break
    else:
        raise Exception('%s not factorable ' % n)

    m = n/p
    Fout=[]
    for q in range(p): # 0,1
        fp = f[q::p]  # every p'th time sample
        Fp = fft( fp ,inv)
        Fout.extend( Fp )

    for u in range(m):
        scratch = Fout[u::m] # u to end in strides of m
        for q1 in range(p):
            k = q1*m + u  # indices to Fout above that became scratch
            Fout[ k ] = scratch[0] # cuz e**0==1 in loop below
            for q in range(1,p):
                if inv:
                    t = e ** ( j*2*pi*k*q/n )
                else:                    
                    t = e ** ( -j*2*pi*k*q/n )
                Fout[ k ] += scratch[q] * t

    return Fout

def rifft(F):
    N = len(F) - 1
    Z = [0] * (N)
    for k in range(N):
        Fek = ( F[k] + F[-k-1].conjugate() )
        Fok = ( F[k] - F[-k-1].conjugate() ) * e ** (j*pi*k/N)
        Z[k] = Fek + j*Fok

    fp = fft(Z , 1)

    f = []
    for c in fp:
        f.append(c.real)
        f.append(c.imag)
    return f

def real_fft( f,inv ):
    if inv:
        return rifft(f)

    N = len(f) / 2

    res = f[::2]
    ims = f[1::2]

    fp = [ complex(r,i) for r,i in zip(res,ims) ]
    print 'fft input ', fp
    Fp = fft( fp ,0 )
    print 'fft output ', Fp

    F = [ complex(0,0) ] * ( N+1 )
    
    F[0] = complex( Fp[0].real + Fp[0].imag , 0 ) 

    for k in range(1,N/2+1):
        tw = e ** ( -j*pi*(.5+float(k)/N ) )
        
        F1k = Fp[k] + Fp[N-k].conjugate()
        F2k = Fp[k] - Fp[N-k].conjugate()
        F2k *= tw
        F[k] = ( F1k + F2k ) * .5
        F[N-k] = ( F1k - F2k ).conjugate() * .5
        #F[N-k] = ( F1kp + e ** ( -j*pi*(.5+float(N-k)/N ) ) * F2kp ) * .5
        #F[N-k] = ( F1k.conjugate() - tw.conjugate() * F2k.conjugate() ) * .5

    F[N] = complex( Fp[0].real - Fp[0].imag , 0 ) 
    return F

def main():
    #fft_func = fft
    fft_func = real_fft

    tvec = [0.309655,0.815653,0.768570,0.591841,0.404767,0.637617,0.007803,0.012665]
    Ftvec = [ complex(r,i) for r,i in zip(
                [3.548571,-0.378761,-0.061950,0.188537,-0.566981,0.188537,-0.061950,-0.378761],
                [0.000000,-1.296198,-0.848764,0.225337,0.000000,-0.225337,0.848764,1.296198] ) ]

    F = fft_func( tvec,0 )

    nerrs= 0
    for i in range(len(Ftvec)/2 + 1):
        if abs( F[i] - Ftvec[i] )> 1e-5:
            print 'F[%d]: %s != %s' % (i,F[i],Ftvec[i])
            nerrs += 1

    print '%d errors in forward fft' % nerrs
    if nerrs:
        return

    trec = fft_func( F , 1 )

    for i in range(len(trec) ):
        trec[i] /= len(trec)

    for i in range(len(tvec) ):
        if abs( trec[i] - tvec[i] )> 1e-5:
            print 't[%d]: %s != %s' % (i,tvec[i],trec[i])
            nerrs += 1

    print '%d errors in reverse fft' % nerrs


def make_random(dims=[1]):
    import Numeric 
    res = []
    for i in range(dims[0]):
        if len(dims)==1:
            r=random.uniform(-1,1)
            i=random.uniform(-1,1)
            res.append( complex(r,i) )
        else:
            res.append( make_random( dims[1:] ) )
    return Numeric.array(res)

def flatten(x):
    import Numeric
    ntotal = Numeric.product(Numeric.shape(x))
    return Numeric.reshape(x,(ntotal,))

def randmat( ndims ):
    dims=[]
    for i in range( ndims ):
        curdim = int( random.uniform(2,4) )
        dims.append( curdim )
    return make_random(dims )

def test_fftnd(ndims=3):
    import FFT
    import Numeric

    x=randmat( ndims )
    print 'dimensions=%s' % str( Numeric.shape(x) )
    #print 'x=%s' %str(x)
    xver = FFT.fftnd(x)
    x2=myfftnd(x)
    err = xver - x2
    errf = flatten(err)
    xverf = flatten(xver)
    errpow = Numeric.vdot(errf,errf)+1e-10
    sigpow = Numeric.vdot(xverf,xverf)+1e-10
    snr = 10*math.log10(abs(sigpow/errpow) )
    if snr<80:
        print xver
        print x2
    print 'SNR=%sdB' % str( snr )
 
def myfftnd(x):
    import Numeric
    xf = flatten(x)
    Xf = fftndwork( xf , Numeric.shape(x) )
    return Numeric.reshape(Xf,Numeric.shape(x) )

def fftndwork(x,dims):
    import Numeric
    dimprod=Numeric.product( dims )

    for k in range( len(dims) ):
        cur_dim=dims[ k ]
        stride=dimprod/cur_dim
        next_x = [complex(0,0)]*len(x)
        for i in range(stride):
            next_x[i*cur_dim:(i+1)*cur_dim] = fft(x[i:(i+cur_dim)*stride:stride],0)
        x = next_x
    return x

if __name__ == "__main__":
    try:
        nd = int(sys.argv[1])
    except:
        nd=None
    if nd:    
        test_fftnd( nd )
    else:    
        sys.exit(0)
