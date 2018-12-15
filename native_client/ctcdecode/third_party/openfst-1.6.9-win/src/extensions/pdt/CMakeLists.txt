file(GLOB HEADER_FILES ../../include/fst/extensions/pdt/*.h)
message(STATUS "${HEADER_FILES}")

if(HAVE_SCRIPT)
  add_library(fstpdtscript getters.cc pdtscript.cc ${HEADER_FILES})
  target_link_libraries(fstpdtscript fstscript fst)
  set_target_properties(fstpdtscript PROPERTIES 
    SOVERSION "${SOVERSION}"
    FOLDER pdt
  )

  install(TARGETS fstpdtscript
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION lib
  )
endif(HAVE_SCRIPT)

if(HAVE_BIN)
  function (add_executable2 _name)
      add_executable(${ARGV})
      if (TARGET ${_name})
          target_link_libraries(${_name} fstpdtscript fstscript fst ${CMAKE_DL_LIBS})
          set_target_properties(${_name} PROPERTIES
            FOLDER pdt/bin
          )
      endif()
      install(TARGETS ${_name} RUNTIME DESTINATION bin)
  endfunction()

  add_executable2(pdtcompose  pdtcompose.cc)
  add_executable2(pdtexpand  pdtexpand.cc)
  add_executable2(pdtinfo  pdtinfo.cc)
  add_executable2(pdtreplace  pdtreplace.cc)
  add_executable2(pdtreverse  pdtreverse.cc)
  add_executable2(pdtshortestpath  pdtshortestpath.cc)
endif(HAVE_BIN)
