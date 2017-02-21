#ifndef KFC_H
#define KFC_H
#include "kiss_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
KFC -- Kiss FFT Cache

Not needing to deal with kiss_fft_alloc and a config 
object may be handy for a lot of programs.

KFC uses the underlying KISS FFT functions, but caches the config object. 
The first time kfc_fft or kfc_ifft for a given FFT size, the cfg 
object is created for it.  All subsequent calls use the cached 
configuration object.

NOTE:
You should probably not use this if your program will be using a lot 
of various sizes of FFTs.  There is a linear search through the
cached objects.  If you are only using one or two FFT sizes, this
will be negligible. Otherwise, you may want to use another method 
of managing the cfg objects.
 
 There is no automated cleanup of the cached objects.  This could lead 
to large memory usage in a program that uses a lot of *DIFFERENT* 
sized FFTs.  If you want to force all cached cfg objects to be freed,
call kfc_cleanup.
 
 */

/*forward complex FFT */
void kfc_fft(int nfft, const kiss_fft_cpx * fin,kiss_fft_cpx * fout);
/*reverse complex FFT */
void kfc_ifft(int nfft, const kiss_fft_cpx * fin,kiss_fft_cpx * fout);

/*free all cached objects*/
void kfc_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif
