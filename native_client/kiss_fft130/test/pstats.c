#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include <sys/types.h>
#include <unistd.h>

#include "pstats.h"

static struct tms tms_beg;
static struct tms tms_end;
static int has_times = 0;


void pstats_init(void)
{
    has_times = times(&tms_beg) != -1;
}

static void tms_report(void)
{
    double cputime;
    if (! has_times )
        return;
    times(&tms_end);
    cputime = ( ((float)tms_end.tms_utime + tms_end.tms_stime + tms_end.tms_cutime + tms_end.tms_cstime ) -
                ((float)tms_beg.tms_utime + tms_beg.tms_stime + tms_beg.tms_cutime + tms_beg.tms_cstime ) )
               / sysconf(_SC_CLK_TCK);
    fprintf(stderr,"\tcputime=%.3f\n" , cputime);
}

static void ps_report(void)
{
    char buf[1024];
#ifdef __APPLE__ /*  MAC OS X */
    sprintf(buf,"ps -o command,majflt,minflt,rss,pagein,vsz -p %d 1>&2",getpid() );
#else /* GNU/Linux */
    sprintf(buf,"ps -o comm,majflt,minflt,rss,drs,pagein,sz,trs,vsz %d 1>&2",getpid() );
#endif    
    if (system( buf )==-1) {
        perror("system call to ps failed");
    }
}

void pstats_report()
{
    ps_report();
    tms_report();
}

