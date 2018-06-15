# This CMake file was created by Lane Schwartz <dowobeha@gmail.com>

# Explicitly list the source files for this subdirectory
#
# If you add any source files to this subdirectory
#    that should be included in the kenlm library,
#        (this excludes any unit test files)
#    you should add them to the following list:
#
# In order to allow CMake files in the parent directory
#    to see this variable definition, we set PARENT_SCOPE.
#
# In order to set correct paths to these files
#    when this variable is referenced by CMake files in the parent directory,
#    we prefix all files with ${CMAKE_CURRENT_SOURCE_DIR}.
#
set(KENLM_UTIL_DOUBLECONVERSION_SOURCE
		${CMAKE_CURRENT_SOURCE_DIR}/bignum-dtoa.cc
		${CMAKE_CURRENT_SOURCE_DIR}/bignum.cc
		${CMAKE_CURRENT_SOURCE_DIR}/cached-powers.cc
		${CMAKE_CURRENT_SOURCE_DIR}/diy-fp.cc
		${CMAKE_CURRENT_SOURCE_DIR}/double-conversion.cc
		${CMAKE_CURRENT_SOURCE_DIR}/fast-dtoa.cc
		${CMAKE_CURRENT_SOURCE_DIR}/fixed-dtoa.cc
		${CMAKE_CURRENT_SOURCE_DIR}/strtod.cc
	PARENT_SCOPE)

