# Helper functions used across the CMake build system

include(CMakeParseArguments)

# Adds a bunch of executables to the build, each depending on the specified
# dependent object files and linking against the specified libraries
function(AddExes)
  set(multiValueArgs EXES DEPENDS LIBRARIES)
  cmake_parse_arguments(AddExes "" "" "${multiValueArgs}" ${ARGN})

  # Iterate through the executable list
  foreach(exe ${AddExes_EXES})

    # Compile the executable, linking against the requisite dependent object files
    add_executable(${exe} ${exe}_main.cc ${AddExes_DEPENDS})

    # Link the executable against the supplied libraries
    target_link_libraries(${exe} ${AddExes_LIBRARIES})

    # Group executables together
    set_target_properties(${exe} PROPERTIES FOLDER executables)

  # End for loop
  endforeach(exe)

  # Install the executable files
  install(TARGETS ${AddExes_EXES} DESTINATION bin)
endfunction()

# Adds a single test to the build, depending on the specified dependent
# object files, linking against the specified libraries, and with the
# specified command line arguments
function(KenLMAddTest)
  cmake_parse_arguments(KenLMAddTest "" "TEST"
                        "DEPENDS;LIBRARIES;TEST_ARGS" ${ARGN})

  # Compile the executable, linking against the requisite dependent object files
  add_executable(${KenLMAddTest_TEST}
                 ${KenLMAddTest_TEST}.cc
                 ${KenLMAddTest_DEPENDS})

  if (Boost_USE_STATIC_LIBS)
    set(DYNLINK_FLAGS)
  else()
    set(DYNLINK_FLAGS COMPILE_FLAGS -DBOOST_TEST_DYN_LINK)
  endif()

  # Require the following compile flag
  set_target_properties(${KenLMAddTest_TEST} PROPERTIES
                        ${DYNLINK_FLAGS}
                        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/tests)

  target_link_libraries(${KenLMAddTest_TEST} ${KenLMAddTest_LIBRARIES} ${TIMER_LINK})

  set(test_params "")
  if(KenLMAddTest_TEST_ARGS)
    set(test_params ${KenLMAddTest_TEST_ARGS})
  endif()

  # Specify command arguments for how to run each unit test
  #
  # Assuming that foo was defined via add_executable(foo ...),
  #   the syntax $<TARGET_FILE:foo> gives the full path to the executable.
  #
  add_test(NAME ${KenLMAddTest_TEST}
           COMMAND $<TARGET_FILE:${KenLMAddTest_TEST}> ${test_params})

  # Group unit tests together
  set_target_properties(${KenLMAddTest_TEST} PROPERTIES FOLDER "unit_tests")
endfunction()

# Adds a bunch of tests to the build, each depending on the specified
# dependent object files and linking against the specified libraries
function(AddTests)
  set(multiValueArgs TESTS DEPENDS LIBRARIES TEST_ARGS)
  cmake_parse_arguments(AddTests "" "" "${multiValueArgs}" ${ARGN})

  # Iterate through the Boost tests list
  foreach(test ${AddTests_TESTS})
    KenLMAddTest(TEST ${test}
                 DEPENDS ${AddTests_DEPENDS}
                 LIBRARIES ${AddTests_LIBRARIES}
                 TEST_ARGS ${AddTests_TEST_ARGS})
  endforeach(test)
endfunction()
