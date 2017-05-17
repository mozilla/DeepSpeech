
#include <float.h>

/* #undef ENABLE_DOUBLE */

#ifdef ENABLE_DOUBLE
# define csf_float double
# define csf_ceil ceil
# define csf_floor floor
# define csf_sin sin
# define csf_log log
# define csf_log10 log10
# define csf_pow pow
# define csf_sqrt sqrt
# define csf_float_min DBL_MIN
#else
# define csf_float float
# define csf_ceil ceilf
# define csf_floor floorf
# define csf_sin sinf
# define csf_log logf
# define csf_log10 log10f
# define csf_pow powf
# define csf_sqrt sqrtf
# define csf_float_min FLT_MIN
#endif
