#!/usr/bin/env python

import math
import sys
import os
import random
import struct
import popen2
import getopt
import numpy

pi=math.pi
e=math.e
j=complex(0,1)

doreal=0

datatype = os.environ.get('DATATYPE','float')

util = '../tools/fft_' + datatype
minsnr=90
if datatype == 'double':
    fmt='d'
elif datatype=='int16_t':
    fmt='h'
    minsnr=10
elif datatype=='int32_t':
    fmt='i'
elif datatype=='simd':
    fmt='4f'
    sys.stderr.write('testkiss.py does not yet test simd')
    sys.exit(0)
elif datatype=='float':
    fmt='f'
else:
    sys.stderr.write('unrecognized datatype %s\n' % datatype)
    sys.exit(1)
 

def dopack(x,cpx=1):
    x = numpy.reshape( x, ( numpy.size(x),) )
    
    if cpx:
        s = ''.join( [ struct.pack(fmt*2,c.real,c.imag) for c in x ] )
    else:
        s = ''.join( [ struct.pack(fmt,c.real) for c in x ] )
    return s

def dounpack(x,cpx):
    uf = fmt * ( len(x) / struct.calcsize(fmt) )
    s = struct.unpack(uf,x)
    if cpx:
        return numpy.array(s[::2]) + numpy.array( s[1::2] )*j
    else:
        return numpy.array(s )

def make_random(dims=[1]):
    res = []
    for i in range(dims[0]):
        if len(dims)==1:
            r=random.uniform(-1,1)
            if doreal:
                res.append( r )
            else:
                i=random.uniform(-1,1)
                res.append( complex(r,i) )
        else:
            res.append( make_random( dims[1:] ) )
    return numpy.array(res)

def flatten(x):
    ntotal = numpy.size(x)
    return numpy.reshape(x,(ntotal,))

def randmat( ndims ):
    dims=[]
    for i in range( ndims ):
        curdim = int( random.uniform(2,5) )
        if doreal and i==(ndims-1):
            curdim = int(curdim/2)*2 # force even last dimension if real
        dims.append( curdim )
    return make_random(dims )

def test_fft(ndims):
    x=randmat( ndims )


    if doreal:
        xver = numpy.fft.rfftn(x)
    else:
        xver = numpy.fft.fftn(x)
    
    open('/tmp/fftexp.dat','w').write(dopack( flatten(xver) , True ) )

    x2=dofft(x,doreal)
    err = xver - x2
    errf = flatten(err)
    xverf = flatten(xver)
    errpow = numpy.vdot(errf,errf)+1e-10
    sigpow = numpy.vdot(xverf,xverf)+1e-10
    snr = 10*math.log10(abs(sigpow/errpow) )
    print 'SNR (compared to NumPy) : %.1fdB' % float(snr)

    if snr<minsnr:
        print 'xver=',xver
        print 'x2=',x2
        print 'err',err
        sys.exit(1)
 
def dofft(x,isreal):
    dims=list( numpy.shape(x) )
    x = flatten(x)

    scale=1
    if datatype=='int16_t':
        x = 32767 * x
        scale = len(x) / 32767.0
    elif datatype=='int32_t':
        x = 2147483647.0 * x
        scale = len(x) / 2147483647.0

    cmd='%s -n ' % util
    cmd += ','.join([str(d) for d in dims])
    if doreal:
        cmd += ' -R '

    print cmd
    p = popen2.Popen3(cmd )

    open('/tmp/fftin.dat','w').write(dopack( x , isreal==False ) )

    p.tochild.write( dopack( x , isreal==False ) )
    p.tochild.close()

    res = dounpack( p.fromchild.read() , 1 )
    open('/tmp/fftout.dat','w').write(dopack( flatten(res) , True ) )
    if doreal:
        dims[-1] = int( dims[-1]/2 ) + 1

    res = scale * res

    p.wait()
    return numpy.reshape(res,dims)

def main():
    opts,args = getopt.getopt(sys.argv[1:],'r')
    opts=dict(opts)

    global doreal
    doreal = opts.has_key('-r')

    if doreal:
        print 'Testing multi-dimensional real FFTs'
    else:
        print 'Testing multi-dimensional FFTs'

    for dim in range(1,4):
        test_fft( dim )

if __name__ == "__main__":
    main()

