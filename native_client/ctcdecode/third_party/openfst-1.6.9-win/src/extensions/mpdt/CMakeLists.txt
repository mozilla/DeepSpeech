file(GLOB HEADER_FILES ../../include/fst/extensions/mpdt/*.h)
message(STATUS "${HEADER_FILES}")

if(HAVE_SCRIPT)
  add_library(fstmpdtscript mpdtscript.cc ${HEADER_FILES})
  target_link_libraries(fstmpdtscript fstscript fst)
  set_target_properties(fstmpdtscript PROPERTIES 
    SOVERSION "${SOVERSION}"
    FOLDER mpdt
  )
  install(TARGETS fstmpdtscript 
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION lib
  )
endif(HAVE_SCRIPT)

if(HAVE_BIN)
  function (add_executable2 _name)
      add_executable(${ARGV})
      if (TARGET ${_name})
          target_link_libraries(${_name} fstmpdtscript fstpdtscript fstscript fst ${CMAKE_DL_LIBS})
          set_target_properties(${_name} PROPERTIES
            FOLDER mpdt/bin
          )
      endif()
    install(TARGETS ${_name} RUNTIME DESTINATION bin)
  endfunction()
  add_executable2(mpdtcompose  mpdtcompose.cc)
  add_executable2(mpdtexpand  mpdtexpand.cc)
  add_executable2(mpdtinfo  mpdtinfo.cc)
  add_executable2(mpdtreverse  mpdtreverse.cc)
endif(HAVE_BIN)

