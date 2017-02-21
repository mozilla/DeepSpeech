If you are reading this, it means you think you may be interested in using the SIMD extensions in kissfft 
to do 4 *separate* FFTs at once.

Beware! Beyond here there be dragons!

This API is not easy to use, is not well documented, and breaks the KISS principle.  


Still reading? Okay, you may get rewarded for your patience with a considerable speedup 
(2-3x) on intel x86 machines with SSE if you are willing to jump through some hoops.

The basic idea is to use the packed 4 float __m128 data type as a scalar element.  
This means that the format is pretty convoluted. It performs 4 FFTs per fft call on signals A,B,C,D.

For complex data, the data is interlaced as follows:
rA0,rB0,rC0,rD0,      iA0,iB0,iC0,iD0,   rA1,rB1,rC1,rD1, iA1,iB1,iC1,iD1 ...
where "rA0" is the real part of the zeroth sample for signal A

Real-only data is laid out:
rA0,rB0,rC0,rD0,     rA1,rB1,rC1,rD1,      ... 

Compile with gcc flags something like
-O3 -mpreferred-stack-boundary=4  -DUSE_SIMD=1 -msse 

Be aware of SIMD alignment.  This is the most likely cause of segfaults.  
The code within kissfft uses scratch variables on the stack.  
With SIMD, these must have addresses on 16 byte boundaries.  
Search on "SIMD alignment" for more info.



Robin at Divide Concept was kind enough to share his code for formatting to/from the SIMD kissfft.  
I have not run it -- use it at your own risk.  It appears to do 4xN and Nx4 transpositions 
(out of place).

void SSETools::pack128(float* target, float* source, unsigned long size128)
{
   __m128* pDest = (__m128*)target;
   __m128* pDestEnd = pDest+size128;
   float* source0=source;
   float* source1=source0+size128;
   float* source2=source1+size128;
   float* source3=source2+size128;

   while(pDest<pDestEnd)
   {
       *pDest=_mm_set_ps(*source3,*source2,*source1,*source0);
       source0++;
       source1++;
       source2++;
       source3++;
       pDest++;
   }
}

void SSETools::unpack128(float* target, float* source, unsigned long size128)
{

   float* pSrc = source;
   float* pSrcEnd = pSrc+size128*4;
   float* target0=target;
   float* target1=target0+size128;
   float* target2=target1+size128;
   float* target3=target2+size128;

   while(pSrc<pSrcEnd)
   {
       *target0=pSrc[0];
       *target1=pSrc[1];
       *target2=pSrc[2];
       *target3=pSrc[3];
       target0++;
       target1++;
       target2++;
       target3++;
       pSrc+=4;
   }
} 
