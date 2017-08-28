/*
POSIX getopt for Windows

AT&T Public License

Code given out at the 1985 UNIFORUM conference in Dallas.
*/

#ifdef __GNUC__
#include <getopt.h>
#endif
#ifndef __GNUC__

#ifndef UTIL_GETOPT_H
#define UTIL_GETOPT_H

#ifdef __cplusplus
extern "C" {
#endif

extern int opterr;
extern int optind;
extern int optopt;
extern char *optarg;
extern int getopt(int argc, char **argv, char *opts);

#ifdef __cplusplus
}
#endif

#endif  /* UTIL_GETOPT_H */
#endif  /* __GNUC__ */

