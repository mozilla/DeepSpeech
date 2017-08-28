# This CMake file was created by Lane Schwartz <dowobeha@gmail.com>

# Explicitly list the source files for this subdirectory
#
# If you add any source files to this subdirectory
#    that should be included in the kenlm library,
#        (this excludes any unit test files)
#    you should add them to the following list:
#
# In order to set correct paths to these files
#    in case this variable is referenced by CMake files in the parent directory,
#    we prefix all files with ${CMAKE_CURRENT_SOURCE_DIR}.
#
set(KENLM_LM_COMMON_SOURCE
		${CMAKE_CURRENT_SOURCE_DIR}/model_buffer.cc
		${CMAKE_CURRENT_SOURCE_DIR}/print.cc
		${CMAKE_CURRENT_SOURCE_DIR}/renumber.cc
		${CMAKE_CURRENT_SOURCE_DIR}/size_option.cc
  PARENT_SCOPE)

if(BUILD_TESTING)
  KenLMAddTest(TEST model_buffer_test
               LIBRARIES kenlm
               TEST_ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test_data)
endif()
