
if(HAVE_COMPACT)
  add_subdirectory(compact)
endif(HAVE_COMPACT)

if(HAVE_COMPRESS)
  add_subdirectory(compress)
endif(HAVE_COMPRESS)

if(HAVE_CONST)
  add_subdirectory(const)
endif(HAVE_CONST)

if(HAVE_FAR OR HAVE_GRM)
  add_subdirectory(far)
endif(HAVE_FAR OR HAVE_GRM)

if(HAVE_LINEAR)
  add_subdirectory(linear)
endif(HAVE_LINEAR)

if(HAVE_LOOKAHEAD)
  add_subdirectory(lookahead)
endif(HAVE_LOOKAHEAD)

if(HAVE_MPDT OR HAVE_GRM)
  add_subdirectory(mpdt)
endif(HAVE_MPDT OR HAVE_GRM)

if(HAVE_NGRAM)
  add_subdirectory(ngram)
endif(HAVE_NGRAM)

#if(HAVE_PYTHON)
#  add_subdirectory(far)
#  add_subdirectory(python)
#endif(HAVE_PDT)
#
if(HAVE_PDT OR HAVE_MPDT OR HAVE_GRM)
  add_subdirectory(pdt)
endif(HAVE_PDT OR HAVE_MPDT OR HAVE_GRM)

if(HAVE_SPECIAL)
  add_subdirectory(special)
endif(HAVE_SPECIAL)

