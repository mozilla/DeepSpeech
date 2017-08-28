find_package(Eigen3)

if(EIGEN3_FOUND)
  if (3.1.0 VERSION_LESS ${EIGEN3_VERSION})
    include_directories(${EIGEN3_INCLUDE_DIR})

    set(KENLM_INTERPOLATE_SOURCE
        backoff_reunification.cc
        bounded_sequence_encoding.cc
        merge_probabilities.cc
        merge_vocab.cc
        normalize.cc
        pipeline.cc
        split_worker.cc
        tune_derivatives.cc
        tune_instances.cc
        tune_weights.cc
        universal_vocab.cc)
  
    find_package(OpenMP)
    if (OPENMP_FOUND)
      set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    else()
      message(STATUS "OpenMP support would be nice for parallelizing matrix operations.")
    endif()
  
    add_library(kenlm_interpolate ${KENLM_INTERPOLATE_SOURCE})
  
    set(KENLM_INTERPOLATE_EXES
        interpolate
        streaming_example)
		
	if (NOT MSVC)
	  set(THREADS pthread)
	endif()
  
      set(KENLM_INTERPOLATE_LIBS
        kenlm_interpolate kenlm kenlm_util ${Boost_LIBRARIES} ${THREADS})
  
    AddExes(EXES ${KENLM_INTERPOLATE_EXES}
            LIBRARIES ${KENLM_INTERPOLATE_LIBS})
  
    if(BUILD_TESTING)
        AddTests(TESTS backoff_reunification_test bounded_sequence_encoding_test normalize_test tune_derivatives_test
                 LIBRARIES ${KENLM_INTERPOLATE_LIBS} pthread)
  
        # tune_instances_test needs an extra command line parameter
        KenLMAddTest(TEST tune_instances_test
                     LIBRARIES ${KENLM_INTERPOLATE_LIBS}
                     TEST_ARGS -- ${CMAKE_CURRENT_SOURCE_DIR}/../common/test_data/toy0.1)
  
        foreach(test_file test1 test2 test3 test_bad_order test_no_unk)
          set(KENLM_MERGE_TESTS_PATH ${KENLM_MERGE_TESTS_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/merge_test/${test_file})
        endforeach(test_file)
  
        KenLMAddTest(TEST merge_vocab_test
                     LIBRARIES ${KENLM_INTERPOLATE_LIBS}
                     TEST_ARGS ${KENLM_MERGE_TESTS_PATH})
    endif()
  else()
    message(WARNING "Not building interpolation. You have an old version of Eigen3, ${EIGEN3_VERSION}, which has a race condition: http://eigen.tuxfamily.org/bz/show_bug.cgi?id=466.  Please install Eigen 3.1.0 or above.")
  endif()
else()
  message(WARNING "Not building interpolation. Eigen3 was not found.")
  message(STATUS "To install Eigen3 in your home directory, copy paste this:\n"
      "export EIGEN3_ROOT=$HOME/eigen-eigen-07105f7124f9\n"
      "(cd $HOME; wget -O - https://bitbucket.org/eigen/eigen/get/3.2.8.tar.bz2 |tar xj)\n"
      "rm CMakeCache.txt\n")
endif()
