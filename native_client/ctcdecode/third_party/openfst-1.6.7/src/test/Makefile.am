AM_CPPFLAGS = -I$(srcdir)/../include $(ICU_CPPFLAGS)
LDADD = ../lib/libfst.la -lm $(DL_LIBS)

check_PROGRAMS = fst_test weight_test

fst_test_SOURCES = fst_test.cc fst_test.h

weight_test_SOURCES = weight_test.cc weight-tester.h

algo_test_SOURCES = algo_test.cc algo_test.h rand-fst.h

check_PROGRAMS += algo_test_log
algo_test_log_SOURCES = $(algo_test_SOURCES)
algo_test_log_CPPFLAGS = -DTEST_LOG $(AM_CPPFLAGS)

check_PROGRAMS += algo_test_tropical
algo_test_tropical_SOURCES = $(algo_test_SOURCES)
algo_test_tropical_CPPFLAGS = -DTEST_TROPICAL $(AM_CPPFLAGS)

check_PROGRAMS += algo_test_minmax
algo_test_minmax_SOURCES = $(algo_test_SOURCES)
algo_test_minmax_CPPFLAGS = -DTEST_MINMAX $(AM_CPPFLAGS)

check_PROGRAMS += algo_test_lexicographic
algo_test_lexicographic_SOURCES = $(algo_test_SOURCES)
algo_test_lexicographic_CPPFLAGS = -DTEST_LEXICOGRAPHIC $(AM_CPPFLAGS)

check_PROGRAMS += algo_test_power
algo_test_power_SOURCES = $(algo_test_SOURCES)
algo_test_power_CPPFLAGS = -DTEST_POWER $(AM_CPPFLAGS)

TESTS = $(check_PROGRAMS)
