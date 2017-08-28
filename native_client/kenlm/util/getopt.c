/*
POSIX getopt for Windows

AT&T Public License

Code given out at the 1985 UNIFORUM conference in Dallas.
*/

#ifndef __GNUC__

#include "getopt.hh"
#include <stdio.h>
#include <string.h>

#define NULL	0
#define EOF	(-1)
#define ERR(s, c)	if(opterr){\
	char errbuf[2];\
	errbuf[0] = c; errbuf[1] = '\n';\
	fputs(argv[0], stderr);\
	fputs(s, stderr);\
	fputc(c, stderr);}
	//(void) write(2, argv[0], (unsigned)strlen(argv[0]));\
	//(void) write(2, s, (unsigned)strlen(s));\
	//(void) write(2, errbuf, 2);}

int	opterr = 1;
int	optind = 1;
int	optopt;
char	*optarg;

int
getopt(argc, argv, opts)
int	argc;
char	**argv, *opts;
{
	static int sp = 1;
	register int c;
	register char *cp;

	if(sp == 1)
		if(optind >= argc ||
		   argv[optind][0] != '-' || argv[optind][1] == '\0')
			return(EOF);
		else if(strcmp(argv[optind], "--") == NULL) {
			optind++;
			return(EOF);
		}
	optopt = c = argv[optind][sp];
	if(c == ':' || (cp=strchr(opts, c)) == NULL) {
		ERR(": illegal option -- ", c);
		if(argv[optind][++sp] == '\0') {
			optind++;
			sp = 1;
		}
		return('?');
	}
	if(*++cp == ':') {
		if(argv[optind][sp+1] != '\0')
			optarg = &argv[optind++][sp+1];
		else if(++optind >= argc) {
			ERR(": option requires an argument -- ", c);
			sp = 1;
			return('?');
		} else
			optarg = argv[optind++];
		sp = 1;
	} else {
		if(argv[optind][++sp] == '\0') {
			sp = 1;
			optind++;
		}
		optarg = NULL;
	}
	return(c);
}

#endif  /* __GNUC__ */
